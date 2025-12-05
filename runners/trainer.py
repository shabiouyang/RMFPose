import sys
import os
import argparse
import pickle
import time
import json
import numpy as np
import torch
import cv2
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ipdb import set_trace
from tqdm import tqdm


# from datasets.datasets_nocs import get_data_loaders_from_cfg, process_batch
from datasets.datasets_omni6dpose import get_data_loaders_from_cfg, process_batch, array_to_SymLabel
from networks.posenet_agent import PoseNet 
from configs.config import get_config
from utils.misc import exists_or_mkdir, get_pose_representation
from utils.genpose_utils import merge_results
from utils.misc import average_quaternion_batch, parallel_setup, parallel_cleanup
from utils.metrics import get_metrics, get_rot_matrix
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image
from utils.transforms import *
from cutoop.utils import draw_3d_bbox
from cutoop.transform import *
from cutoop.data_types import *
from cutoop.eval_utils import *

from torch.cuda.amp import autocast


import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

from itertools import islice

def train_score(cfg, train_loader, val_loader, test_loader, score_agent, teacher_model=None):
	# 初始化 SummaryWriter，将日志存储在 runs/score_agent 目录中
	#writer = SummaryWriter('runs/score_agent')
	# 打开文件以追加模式存储损失信息
	current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	loss_file = open(f'logs/loss_log_{current_datetime}.txt', 'a')



	# # For each batch in the dataloader
	# if cfg.local_rank == 0:  # 仅在主进程上创建 tqdm
	#   pbar = tqdm(train_loader)
	# else:
	#   pbar = None

	# if cfg.local_rank == 0:
	#   train_iter = enumerate(pbar)
	# else:
	#   train_iter = enumerate(train_loader)


	for epoch in range(score_agent.module.clock.epoch, cfg.n_epochs):
		torch.cuda.empty_cache()
		torch.distributed.barrier()  # 同步不同 GPU 进程
		# For each batch in the dataloader
		if cfg.local_rank == 0:  # 仅在主进程上创建 tqdm
			# 使用 islice 从 5400 开始迭代 train_loader
			pbar = tqdm(train_loader)
			train_iter = enumerate(pbar)
		else:
			# 非主进程也使用 islice 从 5400 开始迭代 train_loader
			pbar = None
			train_iter = enumerate(train_loader)

		# 统一的迭代处理
		for i, batch_sample in train_iter:
			# warm up
			if score_agent.module.clock.step < cfg.warmup:
				score_agent.module.update_learning_rate()

			# load data
			batch_sample = process_batch(
				batch_sample=batch_sample,
				device=cfg.device,
				pose_mode=cfg.pose_mode,
				PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS,
			)

			#with autocast():
				# train score or energe without feedback
			losses = score_agent.module.train_func(data=batch_sample, gf_mode='score', teacher_model=teacher_model)
			
			# 仅在主进程上更新 tqdm 和记录损失
			if cfg.local_rank == 0:
				if i % 10 == 0:
					# 记录损失到 TensorBoard
					# for loss_name, loss_value in losses.items():
					#     writer.add_scalar(f'Loss/{loss_name}/train', loss_value.item(), score_agent.module.clock.step)
					# 记录损失到 txt 文件

					formatted_losses = [f"{value.item():.5g}" for key, value in losses.items()]
					total_loss = sum([value.item() for key, value in losses.items()])

					loss_values = [loss_value.item() for loss_value in losses.values()]
					loss_str = ', '.join(f'{value:.6f}' for value in loss_values)
					full_loss_str = f"{total_loss:.6f}, {loss_str}"
					loss_file.write(f"Epoch: {epoch:02d}, Iter: {i:06d}, Losses: [{full_loss_str}]\n")
					loss_file.flush()
					if pbar is not None:
						# update_progress(epoch, i, len(pbar), losses)
						# update_progress(len(pbar), epoch, i, losses)
						#pbar.set_description(f"EPOCH_{epoch}[{i}/{len(train_loader)}][total_loss: {total_loss:.6g}], loss: {[value.item() for key, value in formatted_losses.items()]}]")
						pbar.set_description(f"EPOCH_{epoch}[{i}/{len(train_loader)}][total_loss: {total_loss:.5g}], loss: {formatted_losses}]")
						pbar.refresh()


			score_agent.module.clock.tick()

		loss_file.write("================================================================================\n")
		loss_file.flush()

		# updata learning rate and clock
		# if epoch >= 50 and epoch % 50 == 0:
		score_agent.module.save_ckpt()
		score_agent.module.update_learning_rate()
		score_agent.module.clock.tock()

		# start eval
		# if score_agent.module.clock.epoch % cfg.eval_freq == 0:
		#   data_loaders = [train_loader, val_loader, test_loader]
		#   data_modes = ['train', 'val', 'test']
		#   for i in range(len(data_modes)):
		#       test_batch = next(iter(data_loaders[i]))
		#       data_mode = data_modes[i]
		#       test_batch = process_batch(
		#           batch_sample=test_batch,
		#           device=cfg.device,
		#           pose_mode=cfg.pose_mode,
		#       )
		#       score_agent.module.eval_func(test_batch, data_mode)

		# save (ema) model
	
	# 关闭 SummaryWriter
	#writer.close()
	# 关闭文件
	loss_file.close()


def train_energy(cfg, train_loader, val_loader, test_loader, energy_agent, score_agent=None, ranking=False, distillation=False):
	""" Train score network or energe network without ranking
	Args:
		cfg (dict): config file
		train_loader (torch.utils.data.DataLoader): train dataloader
		val_loader (torch.utils.data.DataLoader): validation dataloader
		energy_agent (torch.nn.Module): energy network with ranking
		score_agent (torch.nn.Module): score network
		ranking (bool): train energy network with ranking or not
	Returns:
	"""
	if ranking is False:
		teacher_model = None if not distillation else score_agent.net
		train_score(cfg, train_loader, val_loader, test_loader, energy_agent, teacher_model)
	else:
		for epoch in range(energy_agent.module.clock.epoch, cfg.n_epochs):
			torch.cuda.empty_cache()
			pbar = tqdm(train_loader)
			for i, batch_sample in enumerate(pbar):
				
				''' warm up '''
				if energy_agent.module.clock.step < cfg.warmup:
					energy_agent.module.update_learning_rate()
					
				''' get data '''
				batch_sample = process_batch(
					batch_sample = batch_sample, 
					device=cfg.device, 
					pose_mode=cfg.pose_mode, 
					PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
				)
				
				''' get pose samples from pretrained score network '''
				pred_pose = score_agent.pred_func(data=batch_sample, repeat_num=5, save_path=None)
				
				''' train energy '''
				losses = energy_agent.module.train_func(data=batch_sample, pose_samples=pred_pose, gf_mode='energy')
				pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
				
				energy_agent.module.clock.tick()
			energy_agent.module.update_learning_rate()
			energy_agent.module.clock.tock()

			''' start eval '''
			if energy_agent.module.clock.epoch % cfg.eval_freq == 0:   
				data_loaders = [train_loader, val_loader, test_loader]    
				data_modes = ['train', 'val', 'test']   
				for i in range(len(data_modes)):
					test_batch = next(iter(data_loaders[i]))
					data_mode = data_modes[i]
					test_batch = process_batch(
						batch_sample=test_batch,
						device=cfg.device,
						pose_mode=cfg.pose_mode,
					)
					
					''' get pose samples from pretrained score network '''
					pred_pose = score_agent.pred_func(data=test_batch, repeat_num=5, save_path=None)
					energy_agent.module.eval_func(test_batch, data_mode, None, 'score')
					energy_agent.module.eval_func(test_batch, data_mode, pred_pose, 'energy')
				
				''' save (ema) model '''
				energy_agent.module.save_ckpt()


def train_scale(cfg, train_loader, val_loader, test_loader, scale_agent, score_agent):
	""" Train scale network
	Args:
		cfg (dict): config file
		train_loader (torch.utils.data.DataLoader): train dataloader
		val_loader (torch.utils.data.DataLoader): validation dataloader
		scale_agent (torch.nn.Module): scale network
		score_agent (torch.nn.Module): score network
	Returns:
	"""
	score_agent.eval()
	
	for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
		''' train '''
		torch.cuda.empty_cache()
		# For each batch in the dataloader
		pbar = tqdm(train_loader)
		for i, batch_sample in enumerate(pbar):
			
			''' warm up'''
			if score_agent.clock.step < cfg.warmup:
				score_agent.update_learning_rate()
				
			''' load data '''
			batch_sample = process_batch(
				batch_sample = batch_sample, 
				device=cfg.device, 
				pose_mode=cfg.pose_mode, 
				PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
			)
			
			''' train scale'''
			with torch.no_grad():
				score_agent.encode_func(data=batch_sample)
			losses = scale_agent.module.train_func(data=batch_sample, gf_mode='scale')
			
			pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
			scale_agent.module.clock.tick()
		
		''' updata learning rate and clock '''
		# if epoch >= 50 and epoch % 50 == 0:
		scale_agent.module.update_learning_rate()
		scale_agent.module.clock.tock()

		''' start eval '''
		if scale_agent.module.clock.epoch % cfg.eval_freq == 0:   
			data_loaders = [train_loader, val_loader, test_loader]    
			data_modes = ['train', 'val', 'test']   
			for i in range(len(data_modes)):
				test_batch = next(iter(data_loaders[i]))
				data_mode = data_modes[i]
				test_batch = process_batch(
					batch_sample=test_batch,
					device=cfg.device,
					pose_mode=cfg.pose_mode,
				)
				with torch.no_grad():
					score_agent.encode_func(data=test_batch)
				scale_agent.module.eval_func(test_batch, data_mode, gf_mode='scale')
				
			''' save (ema) model '''
			scale_agent.module.save_ckpt()


def main():
	# 初始化分布式进程组
	torch.distributed.init_process_group(backend='nccl')
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	# load config
	cfg = get_config()
	cfg.local_rank = local_rank
	
	''' Init data loader '''
	if not (cfg.eval or cfg.pred):
		data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['train', 'val', 'test'])
		train_loader = data_loaders['train_loader']
		val_loader = data_loaders['val_loader']
		test_loader = data_loaders['test_loader']
		# 并行化数据加载器
		train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, \
			shuffle=False, num_workers=train_loader.num_workers, prefetch_factor=2, pin_memory=True, drop_last=True, \
			sampler=torch.utils.data.distributed.DistributedSampler(train_loader.dataset))
		val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, \
			shuffle=False, num_workers=val_loader.num_workers, prefetch_factor=2, pin_memory=True, drop_last=True,\
			sampler=torch.utils.data.distributed.DistributedSampler(val_loader.dataset))
		test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, \
			shuffle=False, num_workers=test_loader.num_workers, prefetch_factor=2, pin_memory=True, drop_last=True, \
			sampler=torch.utils.data.distributed.DistributedSampler(test_loader.dataset))
		print('train_set: ', len(train_loader))
		print('val_set: ', len(val_loader))
		print('test_set: ', len(test_loader))
	else:
		data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['test'])
		test_loader = data_loaders['test_loader']   
		print('test_set: ', len(test_loader))
  
	
	''' Init trianing agent and load checkpoints'''
	if cfg.agent_type == 'score':
		cfg.agent_type = 'score'
		score_agent = PoseNet(cfg)
		score_agent.load_ckpt(model_dir='/home/datasets/GenPose2/results/ScoreNet/ckpt_epoch10.pth', model_path=True, load_model_only=True)
		tr_agent = score_agent
		
	elif cfg.agent_type == 'energy':
		cfg.agent_type = 'energy'
		energy_agent = PoseNet(cfg)
		if cfg.pretrained_score_model_path is not None:
			energy_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
			energy_agent.net.pose_score_net.output_zero_initial()
		if cfg.distillation is True:
			cfg.agent_type = 'score'
			score_agent = PoseNet(cfg)
			score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
			cfg.agent_type = 'energy'
		tr_agent = energy_agent
		
	elif cfg.agent_type == 'energy_with_ranking':
		cfg.agent_type = 'score'
		score_agent = PoseNet(cfg)    
		cfg.agent_type = 'energy'
		energy_agent = PoseNet(cfg)
		score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
		tr_agent = energy_agent
	
	elif cfg.agent_type == 'scale':
		cfg.agent_type = 'score'
		cfg.agent_type = 'score'
		score_agent = PoseNet(cfg)
		score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
		cfg.agent_type = 'scale'
		scale_agent = PoseNet(cfg)
		tr_agent = scale_agent
	else:
		raise NotImplementedError
	
	''' Load checkpoints '''
	if cfg.use_pretrain or cfg.eval or cfg.pred:
		tr_agent.load_ckpt(
			model_dir=(
				cfg.pretrained_score_model_path if cfg.agent_type == 'score' else (
					cfg.pretrained_energy_model_path if cfg.agent_type in ['energy', 'energy_with_ranking']
						else cfg.pretrained_scale_model_path
				)
			), 
			model_path=True, 
			load_model_only=False
		)
	# 使用 DistributedDataParallel 进行分布式训练
	tr_agent = torch.nn.parallel.DistributedDataParallel(tr_agent, device_ids=[local_rank], output_device=local_rank)
		
	''' Start training loop '''
	if cfg.agent_type == 'score':
		train_score(cfg, train_loader, val_loader, test_loader, tr_agent)
	elif cfg.agent_type == 'energy':
		if cfg.distillation:
			train_energy(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent, False, True)
		else:
			train_energy(cfg, train_loader, val_loader, test_loader, tr_agent)
	elif cfg.agent_type == 'energy_with_ranking':
		train_energy(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent, True)
	else:
		train_scale(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent)


if __name__ == '__main__':
	main()