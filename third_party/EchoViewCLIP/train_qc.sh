CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=28504 main_qc.py \
-cfg configs/ultracls/echoviewclip_qc.yaml \
--output output/train_qc