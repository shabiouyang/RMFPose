#!/bin/bash
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc_per_node=1 runners/evaluation_fm.py \
--pretrained_score_model_path /home/datasets/GenPose2/results/ScoreNet2/ckpt_epoch44.pth \
--pretrained_scale_model_path /home/datasets/GenPose2/results/ckpts/ScaleNet/scalenet.pth \
--data_path /home/datasets/Omni6DPose/SOPE/ \
--sampler_mode ode \
--percentage_data_for_test 0.1 \
--batch_size 64 \
--seed 0 \
--result_dir single_fm_ikea \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.0 \
--dino pointwise \
--num_worker 32 \
--real_drop 3 \


