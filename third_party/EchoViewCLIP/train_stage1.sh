CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29264 main_1.py \
-cfg configs/ultracls/echoviewclip_stage1.yaml \
--output output/train_stage1