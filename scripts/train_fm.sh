#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 runners/trainer.py \
--data_path /home/datasets/Omni6DPose/SOPE/ \
--log_dir ScoreNet \
--agent_type score \
--sampler_mode ode \
--sampling_steps 500 \
--eval_freq 1 \
--batch_size 128 \
--n_epochs 50 \
--percentage_data_for_train 1.0 \
--percentage_data_for_test 1.0 \
--percentage_data_for_val 1.0 \
--seed 0 \
--is_train \
--dino pointwise \
--num_worker 32 \