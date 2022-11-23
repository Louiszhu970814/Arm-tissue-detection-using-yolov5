python3 -m torch.distributed.launch --nproc_per_node 3 train.py --data data/arm_tissue.yaml --batch 96 --weights yolov5s.pt --workers 8 --epochs 10 
#python3 train.py --data data/arm_tissue.yaml --batch 16  --weights yolov5s.pt --workers 16 --epochs 6 --device 0 --do-semi
#python3 train.py --data data/arm_tissue.yaml --batch 64 --img-size 640 --device 2 --epochs 20 
#python3 train.py --data data/arm_tissue.yaml --batch 64 --img-size 640 --device 2 --epochs 9


