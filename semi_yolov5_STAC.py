from builtins import any, enumerate, hasattr, int, isinstance, len, max, open, range, round, vars
import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import PIL.Image as Image
import torchvision

import test
from models.experimental import attempt_load 
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, LoadImagesAndLabels
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, non_max_suppression, xyxy2xywh
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel


logger = logging.getLogger

class semiDataset(Dataset):
    def __init__(self, imgs, target, path):
        self.imgs = imgs
        self.target = target
        self.path = path
        self.size = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.target[index], self.path[index], self.size


def train(hyp, opt, device):
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
    Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    do_semi = opt.do_semi
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    
    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve #create plots
    cuda = device.type != 'cpu'
    init_seeds(2+rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    nc = 1 if opt.single_cls else int(data_dict['nc'])  #number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)
    
    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)     # download if not found locally
        ckpt = torch.load(weights, map_location=device)   #load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  #create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  #exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude) #intersect
        model.load_state_dict(state_dict, strict=False)  #load
        
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict) #check
    train_path = data_dict['train']
    test_path = data_dict['val']



    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)    # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs

    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)   # no decay 
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply dacay


    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))    # adjust betal to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay':hyp['weight_decay']})    # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})    # add pg2 (biases)
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']   # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)   
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    
    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0

    if pretrained:
        # optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt
        

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weight, epochs)
        if epochs < start_epoch:
            epochs += ckpt['epoch']
        del ckpt, state_dict


        # Image sizes
        gs = max(int(model.stride.max()), 32)     # grid size (max stride)
        nl = model.model[-1].nl   # number of detection layer (used for scaling hyp['obj])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]   # verify imgsz are gs-multiples
        

        # DP mode 
        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # SyncBatchNorm
        if opt.sync_bn and cuda and rank != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        
        # Trainloader
    if do_semi:
        dataloader, dataset, unlabeldataloader = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                    hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                    world_size=opt.world_size, workers=opt.workers,
                                                    image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '), do_semi=opt.do_semi)
    else:
        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '), do_semi=opt.do_semi)


    # Train teacher model
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()      # max label class
    nb = len(dataloader)     # number of batches


    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc-1)

    # process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '), do_semi=False)[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])    # classes
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()   # pre-reduce anchor precision


    # DDP mode
    if cuda and rank!=1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))


    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names


    # Train teacher model --> burn in
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    burnin_epochs = epochs / 2

    # burn in
    for epoch in range(start_epoch, burnin_epochs):   # epoch-------------------------
        model.train()
        nb = len(dataloader)
        mloss = torch.zeros(4, device=device)    # mean loss
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            
            # Warm up 
            if ni <= [0, nw]:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size].round()))
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            # Forward 
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_item = compute_loss(pred, targets.to(device))      # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between device in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward 
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
            
            # print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_item) / (i+1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 compute_loss=compute_loss)


        fi = fitness(np.array(results).reshape(1, -1))           # weighted combination of [P, R, mAP@50, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict()}
            if best_fitness == fi:
                    torch.save(ckpt, best)
            del ckpt
    
        # end epoch ----------------------------------------------------------------------------
    # end warm up

    # get persudo label
    # STAC 
    # first apply weak augmentation on unlabeled dataset then use teacher net to predict the persudo labels 
    # Then apply strong augmentation on unlabeled dataset, use student net to get the logists and compute the unlabeled loss.


    model.eval()
    img = []
    target = []
    Path = []
    imgsz = opt.img_size
    for idx, batch in tqdm (enumerate(unlabeldataloader), total=len(unlabeldataloader)):
        imgs0, _, path, _ = batch  # from uint8 to float16

        with torch.no_grad():
            pred = model(imgs0.to(device, non_blocking=True).float() / 255.0)[0]

        gn = torch.tensor(imgs0.shape)[[3, 2, 3, 2]]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for index, pre in enumerate(pred):
            predict_number = len(pre)
            if predict_number==0:
                continue
            Class = pre[:,5].view(predict_number,1).cpu()
            XYWH = (xyxy2xywh(pre[:,:4])).cpu()
            XYWH/= gn     
            pre = torch.cat((torch.zeros(predict_number,1),Class,XYWH), dim=1)    
            img.append(imgs0[index])
            target.append(pre)
            Path.append(path[index])
    
    unlabeldataset = semiDataset(img, target, Path)
    del img, targets, Path
    model.train()

    


            


