import sys
import os
import torch
import numpy as np
from torch import nn, Tensor

from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from networks.decoder_head.rot_head import RotHead
from networks.decoder_head.trans_head import TransHead
import torch.nn.functional as F
from networks.gf_algorithms.vn_layers import VNLinearLeakyReLU, VNLinear

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# class PoseScoreNet(nn.Module):
#     def __init__(self, marginal_prob_func, dino_dim, pose_mode='quat_wxyz', regression_head='RT', per_point_feature=False):
#         """_summary_

#         Args:
#             marginal_prob_func (func): marginal_prob_func of score network
#             pose_mode (str, optional): the type of pose representation from {'quat_wxyz', 'quat_xyzw', 'rot_matrix', 'euler_xyz'}. Defaults to 'quat_wxyz'.
#             regression_head (str, optional): _description_. Defaults to 'RT'.

#         Raises:
#             NotImplementedError: _description_
#         """
#         super(PoseScoreNet, self).__init__()
#         self.regression_head = regression_head
#         self.per_point_feature = per_point_feature
#         self.dino_dim = dino_dim
#         self.act = nn.ReLU(True)
#         pose_dim = get_pose_dim(pose_mode)
#         ''' encode pose '''
#         self.pose_encoder = nn.Sequential(
#             nn.Linear(pose_dim, 256),
#             self.act,
#             nn.Linear(256, 256),
#             self.act,
#         )
		
#         ''' encode t '''
#         self.t_encoder = nn.Sequential(
#             GaussianFourierProjection(embed_dim=128),
#             # self.act, # M4D26 update
#             nn.Linear(128, 128),
#             self.act,
#         )

#         ''' fusion tail '''
#         if self.regression_head == 'RT':
#             self.fusion_tail = nn.Sequential(
#                 nn.Linear(128+256+1024+dino_dim, 512),
#                 self.act,
#                 zero_module(nn.Linear(512, pose_dim)),
#             )

#         elif self.regression_head == 'R_and_T':
#             ''' rotation regress head '''
#             self.fusion_tail_rot = nn.Sequential(
#                 nn.Linear(128+256+1024+dino_dim, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
#                 self.act,
#                 zero_module(nn.Linear(256, pose_dim - 3)),
#             )
			
#             ''' tranalation regress head '''
#             self.fusion_tail_trans = nn.Sequential(
#                 nn.Linear(128+256+1024+dino_dim, 256),
#                 # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
#                 self.act,
#                 zero_module(nn.Linear(256, 3)),
#             )
			
#         elif self.regression_head == 'Rx_Ry_and_T':
#             if pose_mode != 'rot_matrix':
#                 raise NotImplementedError
#             if per_point_feature:
#                 self.fusion_tail_rot_x = RotHead(in_feat_dim=128+256+1280, out_dim=3)       # t_feat + pose_feat + pts_feat
#                 self.fusion_tail_rot_y = RotHead(in_feat_dim=128+256+1280, out_dim=3)       # t_feat + pose_feat + pts_feat
#                 self.fusion_tail_trans = TransHead(in_feat_dim=128+256+1280, out_dim=3)     # t_feat + pose_feat + pts_feat
#             else:
#                 ''' rotation_x_axis regress head '''
#                 self.fusion_tail_rot_x = nn.Sequential(
#                     nn.Linear(128+256+1024+dino_dim, 256),
#                     # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
#                     self.act,
#                     zero_module(nn.Linear(256, 3)),
#                 )
#                 self.fusion_tail_rot_y = nn.Sequential(
#                     nn.Linear(128+256+1024+dino_dim, 256),
#                     # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
#                     self.act,
#                     zero_module(nn.Linear(256, 3)),
#                 )
				
#                 ''' tranalation regress head '''
#                 self.fusion_tail_trans = nn.Sequential(
#                     nn.Linear(128+256+1024+dino_dim, 256),
#                     # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
#                     self.act,
#                     zero_module(nn.Linear(256, 3)),
#                 )            
#         else:
#             raise NotImplementedError
			
#         self.marginal_prob_func = marginal_prob_func


#     def forward(self, data):
#         '''
#         Args:
#             data, dict {
#                 'pts_feat': [bs, c]
#                 'rgb_feat': [bs, dino_dim] (optional)
#                 'pose_sample': [bs, pose_dim]
#                 't': [bs, 1]
#             }
#         '''
		
#         pts_feat = data['pts_feat']
#         if self.dino_dim:
#             rgb_feat = data['rgb_feat']
#         sampled_pose = data['sampled_pose']
#         t = data['t']
		
#         # pts = pts.permute(0, 2, 1) # -> (bs, 3, 1024)
#         # pts_feat = self.pts_encoder(pts) 
		
#         t_feat = self.t_encoder(t.squeeze(1))
#         pose_feat = self.pose_encoder(sampled_pose)

#         if self.per_point_feature:
#             assert 0
#             num_pts = pts_feat.shape[-1]
#             t_feat = t_feat.unsqueeze(-1).repeat(1, 1, num_pts)
#             pose_feat = pose_feat.unsqueeze(-1).repeat(1, 1, num_pts)
#             total_feat = torch.cat([pts_feat, t_feat, pose_feat], dim=1)
#         else:
#             if self.dino_dim:
#                 total_feat = torch.cat([pts_feat, t_feat, pose_feat, rgb_feat], dim=-1)
#             else:
#                 total_feat = torch.cat([pts_feat, t_feat, pose_feat], dim=-1)
#         _, std = self.marginal_prob_func(total_feat, t)
		
#         if self.regression_head == 'RT':
#             out_score = self.fusion_tail(total_feat) / (std+1e-7) # normalisation
#         elif self.regression_head == 'R_and_T':
#             rot = self.fusion_tail_rot(total_feat)
#             trans = self.fusion_tail_trans(total_feat)
#             out_score = torch.cat([rot, trans], dim=-1) / (std+1e-7) # normalisation
#         elif self.regression_head == 'Rx_Ry_and_T':
#             rot_x = self.fusion_tail_rot_x(total_feat)
#             rot_y = self.fusion_tail_rot_y(total_feat)
#             trans = self.fusion_tail_trans(total_feat)
#             out_score = torch.cat([rot_x, rot_y, trans], dim=-1) / (std+1e-7) # normalisation
#         elif self.regression_head == 'angle_vec_and_T':
#             angle = self.fusion_tail_angle(total_feat)
#             vec = self.fusion_tail_vec(total_feat)
#             trans = self.fusion_tail_trans(total_feat)
#             out_score = torch.cat([angle, vec, trans], dim=-1) / (std+1e-7) # normalisation
#         else:
#             raise NotImplementedError
#         # set_trace()
#         return out_score







# import torch
# import torch.nn as nn
# from einops import rearrange

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model=256, nhead=8, dim_feedforward=512):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.GELU(),
#             nn.Linear(dim_feedforward, d_model)
#         )
#         #self.norm2 = nn.LayerNorm(d_model)
		
#     def forward(self, query, key_value):
#         # 交叉注意力（Query来自条件参数）
#         attn_output, _ = self.cross_attn(
#             query=query,
#             key=key_value,
#             value=key_value
#         )
#         query = self.norm1(query + attn_output)
		
#         # 前馈网络
#         ffn_output = self.ffn(query)
#         #query = self.norm2(query + ffn_output)
#         return ffn_output

# class PoseScoreNet(nn.Module):
#     def __init__(self, marginal_prob_func, dino_dim, pose_mode='quat_wxyz', regression_head='RT', per_point_feature=False):
#         super().__init__()
#         self.pose_dim = 6  # 最终输出维度（3D角速度 + 3D线速度）
		
#         # 点云特征编码（64->256）
#         self.point_encoder = nn.Sequential(
#             nn.Linear(64, 256),
#             nn.LayerNorm(256),
#             nn.GELU()
#         )
		
#         # 时间+位姿条件编码
#         self.condition_encoder = nn.Sequential(
#             nn.Linear(6+1, 128),  # 6D位姿 + 时间t
#             nn.GELU(),
#             nn.Linear(128, 256)
#         )
		
#         # 3层残差Transformer
#         self.transformer_layers = nn.ModuleList([
#             TransformerEncoderLayer() for _ in range(3)
#         ])
		
#         # 6D向量场预测
#         self.vector_head = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.GELU(),
#             zero_module(nn.Linear(128, 6)) 
#         )

#     def forward(self, data):
#         # 输入预处理
#         point_feat = data['pts_feat']  # [bs,64,1024]
#         pose_t = data['sampled_pose']  # [bs,6]
#         t = data['t']                  # [bs,1]
		
#         # 点云特征编码
#         point_feat = self.point_encoder(point_feat.permute(0, 2, 1))  # [bs,1024,256]
		
#         # 条件参数编码
#         condition = torch.cat([pose_t, t], dim=-1)
#         condition = self.condition_encoder(condition)  # [bs,256]
#         condition = rearrange(condition, 'b d -> b 1 d')  # [bs,1,256]
		
#         # Transformer处理
#         for layer in self.transformer_layers:
#             condition = layer(condition, point_feat)
		
#         # 向量场预测
#         vector_field = self.vector_head(condition.squeeze(1))
#         return vector_field







class ContextGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)       # 归一化
        self.linear = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
		
    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate = self.sigmoid(self.linear(x)) # [bs, dim]
        return residual * gate             # 残差门控

class PoseScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, dino_dim, pose_mode='quat_wxyz', regression_head='RT', per_point_feature=False):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network
            pose_mode (str, optional): the type of pose representation from {'quat_wxyz', 'quat_xyzw', 'rot_matrix', 'euler_xyz'}. Defaults to 'quat_wxyz'.
            regression_head (str, optional): _description_. Defaults to 'RT'.

        Raises:
            NotImplementedError: _description_
        """
        super(PoseScoreNet, self).__init__()
        self.pose_dim = 6  # 保持9D姿态表示
        self.dino_dim = dino_dim
		
        ''' 时间编码器（强化时间依赖性） '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 256) 
        )
		
        ''' 姿态编码器（适配Flow Matching特性） '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
		
        ''' 多模态特征融合核心 '''
        self.fusion_core = nn.Sequential(
            nn.Linear(1024 + 256 + 256, 1024),  # pts_feat + pose_feat + t_feat
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            ContextGating(512)  # 增加特征交互门控
        )
		
        ''' 向量场预测头 '''
        self.vector_field_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            zero_module(nn.Linear(256, self.pose_dim)) 
        )
		
    def forward(self, data):
        # 特征提取（保持与原始结构兼容）
        pts_feat = data['pts_feat']        # [bs,1024]
        sampled_pose = data['sampled_pose']# [bs,9] 
        t = data['t']                      # [bs,1]
		
        # 时间特征编码
        t_feat = self.t_encoder(t.squeeze(1))  # [bs,256]
		
        # 姿态编码（关键改进点）
        pose_feat = self.pose_encoder(sampled_pose)  # [bs,256]
		
        # 多模态特征融合
        fused_feat = torch.cat([pts_feat, pose_feat, t_feat], dim=-1)
        fused_feat = self.fusion_core(fused_feat)  # [bs,512]
		
        # 向量场预测（核心输出变更）
        vector_field = self.vector_field_head(fused_feat)  # [bs,9]
        return vector_field


class PoseDecoderNet(nn.Module): # seems useless
    def __init__(self, marginal_prob_func, sigma_data=1.4148, pose_mode='quat_wxyz', regression_head='RT'):
        """_summary_

        Args:
            marginal_prob_func (func): marginal_prob_func of score network
            pose_mode (str, optional): the type of pose representation from {'quat_wxyz', 'quat_xyzw', 'rot_matrix', 'euler_xyz'}. Defaults to 'quat_wxyz'.
            regression_head (str, optional): _description_. Defaults to 'RT'.

        Raises:
            NotImplementedError: _description_
        """
        super(PoseDecoderNet, self).__init__()
        self.sigma_data = sigma_data
        self.regression_head = regression_head
        self.act = nn.ReLU(True)
        pose_dim = get_pose_dim(pose_mode)

        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
		
        ''' encode sigma(t) '''
        self.sigma_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=128),
            nn.Linear(128, 128),
            self.act,
        )

        ''' fusion tail '''
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0) # init the final output layer's weights to zeros

        if self.regression_head == 'RT':
            self.fusion_tail = nn.Sequential(
                nn.Linear(128+256+1024, 512),
                self.act,
                Linear(512, pose_dim, **init_zero),
            )

        elif self.regression_head == 'R_and_T':
            ''' rotation regress head '''
            self.fusion_tail_rot = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, pose_dim - 3, **init_zero),
            )
			
            ''' tranalation regress head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )
			
        elif self.regression_head == 'Rx_Ry_and_T':
            if pose_mode != 'rot_matrix':
                raise NotImplementedError
            ''' rotation_x_axis regress head '''
            self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )
            self.fusion_tail_rot_y = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )
			
            ''' tranalation regress head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                Linear(256, 3, **init_zero),
            )    
		
        else:
            raise NotImplementedError
			
        self.marginal_prob_func = marginal_prob_func


    def forward(self, data):
        '''
        Args:
            data, dict {
                'pts_feat': [bs, c]
                'pose_sample': [bs, pose_dim]
                't': [bs, 1]
            }
        '''

        pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']
        _, sigma_t = self.marginal_prob_func(None, t) # \sigma(t) = t in EDM
		
        # determine scaling functions
        # EDM
        # c_skip = self.sigma_data ** 2 / (sigma_t ** 2 + self.sigma_data ** 2)
        # c_out = self.sigma_data * t / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
        # c_in = 1 / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
        # c_noise = torch.log(sigma_t) / 4
        # VE
        c_skip = 1
        c_out = sigma_t
        c_in = 1 
        c_noise = torch.log(sigma_t / 2)

        # comp total feat 
        sampled_pose_rescale = sampled_pose * c_in
        pose_feat = self.pose_encoder(sampled_pose_rescale)
        sigma_feat = self.sigma_encoder(c_noise.squeeze(1))
        total_feat = torch.cat([pts_feat, sigma_feat, pose_feat], dim=-1)
		
        if self.regression_head == 'RT':
            nn_output = self.fusion_tail(total_feat)
        elif self.regression_head == 'R_and_T':
            rot = self.fusion_tail_rot(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            nn_output = torch.cat([rot, trans], dim=-1)
        elif self.regression_head == 'Rx_Ry_and_T':
            rot_x = self.fusion_tail_rot_x(total_feat)
            rot_y = self.fusion_tail_rot_y(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            nn_output = torch.cat([rot_x, rot_y, trans], dim=-1)
        else:
            raise NotImplementedError
	
        denoised_output = c_skip * sampled_pose + c_out * nn_output
        return denoised_output









