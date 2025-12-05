import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import roma
import numpy as np

from scipy import integrate
from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from utils.misc import normalize_rotation
from copy import deepcopy

from utils.ode_solvers import get_ode_solver
from sklearn.cluster import DBSCAN
import torch
from scipy.spatial.transform import Rotation

from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch


def normalize_rotation_6d(rot_6d):
    """将6D旋转表示归一化为两个正交单位向量，支持batch处理"""
    a1, a2 = rot_6d.split([3, 3], dim=-1)
    a1 = F.normalize(a1, dim=-1)
    b = a2 - torch.sum(a1 * a2, dim=-1, keepdim=True) * a1
    b = F.normalize(b, dim=-1)
    return torch.cat([a1, b], dim=-1)

def rotation_6d_to_matrix(rot_6d):
    """将6D旋转表示转换为旋转矩阵，支持batch处理"""
    rot_6d = normalize_rotation_6d(rot_6d)
    a1, a2 = rot_6d.split([3, 3], dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-1)

def matrix_to_quaternion(matrix):
    """将旋转矩阵转换为四元数，支持batch处理"""
    # 使用scipy的Rotation类进行转换
    batch_size = matrix.shape[0]
    matrix_np = matrix.cpu().numpy()
    
    # 初始化四元数张量
    quat = torch.zeros(batch_size, 4, device=matrix.device, dtype=matrix.dtype)
    
    # 批量转换
    for i in range(batch_size):
        quat_np = Rotation.from_matrix(matrix_np[i]).as_quat()
        quat[i] = torch.tensor(quat_np, dtype=matrix.dtype)
    
    return quat

def quaternion_to_matrix(quat):
    """将四元数转换为旋转矩阵，支持batch处理"""
    # 使用scipy的Rotation类进行转换
    batch_size = quat.shape[0]
    quat_np = quat.cpu().numpy()
    
    # 初始化旋转矩阵张量
    matrix = torch.zeros(batch_size, 3, 3, device=quat.device, dtype=quat.dtype)
    
    # 批量转换
    for i in range(batch_size):
        matrix_np = Rotation.from_quat(quat_np[i]).as_matrix()
        matrix[i] = torch.tensor(matrix_np, dtype=quat.dtype)
    
    return matrix

def quaternion_slerp(q1, q2, t):
    """四元数球面线性插值，支持batch处理"""
    # 确保输入是归一化的
    # q1 = F.normalize(q1, dim=-1)
    # q2 = F.normalize(q2, dim=-1)
    #q2 = q2 - (q1 * q2).sum(-1, keepdim=True) * q1
    #q2 = F.normalize(q2, dim=-1)
    
    # 计算点积
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    
    # 处理方向二义性
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # 防止数值不稳定
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # 计算插值
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # 处理theta接近0的情况（使用线性插值）
    linear_interp = torch.abs(sin_theta) < 1e-6
    
    # 球面插值
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    spherical_result = s0 * q1 + s1 * q2
    
    # 线性插值
    linear_result = (1 - t) * q1 + t * q2
    
    # 根据条件选择结果
    result = torch.where(linear_interp, linear_result, spherical_result)
    
    # 归一化结果
    return F.normalize(result, dim=-1)

def weighted_average_quaternions(quats, weights):
    """加权平均四元数，支持batch处理"""
    # 归一化权重 [B, N]
    weights = F.softmax(weights, dim=-1)
    
    # 选择参考四元数（权重最大的那个）
    max_weight_idx = torch.argmax(weights, dim=-1)  # [B]
    
    # 提取每个batch的参考四元数
    batch_size, num_quats, _ = quats.shape
    batch_indices = torch.arange(batch_size, device=quats.device)
    ref_quats = quats[batch_indices, max_weight_idx]  # [B, 4]
    
    # 确保所有四元数与参考四元数在同一半球
    dot_products = torch.sum(quats * ref_quats.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, 1]
    quats = torch.where(dot_products < 0, -quats, quats)
    
    # 迭代加权平均
    avg_quats = ref_quats
    for _ in range(5):  # 迭代几次以提高精度
        # 计算从当前平均到每个四元数的插值
        t = weights.unsqueeze(-1)  # [B, N, 1]
        interp_quats = quaternion_slerp(
            avg_quats.unsqueeze(1).expand_as(quats), 
            quats, 
            t
        )  # [B, N, 4]
        
        # 加权求和
        avg_quats = torch.sum(interp_quats * weights.unsqueeze(-1), dim=1)  # [B, 4]
        avg_quats = F.normalize(avg_quats, dim=-1)
    
    return avg_quats

def weighted_fusion_poses(poses, weights):
    """
    在四元数空间加权合成6D姿态表示，支持batch处理
    
    参数:
    poses (torch.Tensor): 姿态表示，形状为[bs, num, 9]
    weights (torch.Tensor): 权重，形状为[bs, num]
    
    返回:
    torch.Tensor: 融合后的姿态表示，形状为[bs, 9]
    """
    bs, num, _ = poses.shape
    
    # 分离旋转和平移
    rotations = poses[..., :6]  # [bs, num, 6]
    translations = poses[..., 6:]  # [bs, num, 3]
    
    # 将6D旋转转换为四元数
    rotations_flat = rotations.reshape(bs * num, 6)
    matrices = rotation_6d_to_matrix(rotations_flat)
    quats = matrix_to_quaternion(matrices)
    quats = quats.reshape(bs, num, 4)
    
    # 在四元数空间加权平均
    fused_quat = weighted_average_quaternions(quats, weights)  # [bs, 4]
    
    # 将融合后的四元数转回旋转矩阵
    fused_matrix = quaternion_to_matrix(fused_quat)  # [bs, 3, 3]
    
    # 提取前两列作为6D表示
    fused_rotation = torch.cat([fused_matrix[..., 0], fused_matrix[..., 1]], dim=-1)  # [bs, 6]
    
    # 平移向量的加权平均
    ave_weights = weights / torch.sum(weights, dim=1).unsqueeze(-1)
    fused_translation = torch.sum(translations * ave_weights.unsqueeze(-1), dim=1) # [bs, 3]
    
    # 合并结果
    fused_pose = torch.cat([fused_rotation, fused_translation], dim=-1)  # [bs, 9]
    
    return fused_pose

def SO3_uniform_R3_normal(num_samples, device):
    R = roma.random_rotmat(num_samples).to(device)

    p = torch.randn(num_samples, 3).to(device)

    T = torch.eye(4).repeat(num_samples, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p

    # 计算平移部分的对数概率
    log_prob_translation = -0.5 * (p ** 2).sum(dim=1) - (3/2) * torch.log(torch.tensor(2 * np.pi, device=device))
    
    # 旋转部分的对数概率（常数项）
    #log_prob_rotation_constant = torch.full((num_samples,), -np.log(8 * np.pi**2), device=device)
    
    # 联合对数概率（通常忽略旋转常数项）
    log_prob_joint = log_prob_translation

    return T, log_prob_joint

def global_prior_likelihood(z, sigma_max):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    # z: [bs, pose_dim]
    shape = z.shape
    N = np.prod(shape[1:]) # pose_dim
    return -N / 2. * torch.log(2*np.pi*sigma_max**2) - torch.sum(z**2, dim=-1) / (2 * sigma_max**2)


def cond_ode_likelihood(
        score_model,
        data,
        prior,
        sde_coeff,
        marginal_prob_fn,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        num_steps=None,
        pose_mode='quat_wxyz', 
        init_x=None,
    ):

    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))

    def divergence_eval(data, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        # save ckpt of sampled_pose
        origin_sampled_pose = data['sampled_pose'].clone()
        with torch.enable_grad():
            # make sampled_pose differentiable
            data['sampled_pose'].requires_grad_(True)
            score = score_model(data)
            score_energy = torch.sum(score * epsilon) # [, ]
            grad_score_energy = torch.autograd.grad(score_energy, data['sampled_pose'])[0] # [bs, pose_dim]
        # reset sampled_pose
        data['sampled_pose'] = origin_sampled_pose
        return torch.sum(grad_score_energy * epsilon, dim=-1) # [bs, 1]
    
    def divergence_eval_wrapper(data):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad(): 
            # Compute likelihood.
            div = divergence_eval(data, epsilon) # [bs, 1]
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, inp):        
        """The ODE function for use by the ODE solver."""
        # split x, logp from inp
        x = inp[:-shape[0]]
        logp = inp[-shape[0]:] # haha, actually we do not need use logp here
        # calc x-grad
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        x_grad = drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # calc logp-grad
        logp_grad = drift - 0.5 * (diffusion**2) * divergence_eval_wrapper(data)
        # concat curr grad
        return  np.concatenate([x_grad, logp_grad], axis=0)


    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    epsilon = prior((batch_size, pose_dim)).to(device)
    init_x = data['sampled_pose'].clone().cpu().numpy() if init_x is None else init_x
    shape = init_x.shape
    init_logp = np.zeros((shape[0],)) # [bs]
    init_inp = np.concatenate([init_x.reshape(-1), init_logp], axis=0)
  
    # Run the black-box ODE solver, note the 
    res = integrate.solve_ivp(ode_func, (eps, 1.0), init_inp, rtol=rtol, atol=atol, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device) # [bs * (pose_dim + 1)]
    z = zp[:-shape[0]].reshape(shape) # [bs, pose_dim]
    delta_logp = zp[-shape[0]:].reshape(shape[0]) # [bs,] logp
    _, sigma_max = marginal_prob_fn(None, torch.tensor(1.).to(device)) # we assume T = 1 
    prior_logp = global_prior_likelihood(z, sigma_max)
    log_likelihoods = (prior_logp + delta_logp) / np.log(2) # negative log-likelihoods (nlls)
    return z, log_likelihoods


def cond_pc_sampler(
        score_model, 
        data,
        prior,
        sde_coeff,
        num_steps=500, 
        snr=0.16,                
        device='cuda',
        eps=1e-5,
        pose_mode='quat_wxyz',
        init_x=None,
    ):
    
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim)).to(device) if init_x is None else init_x
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(pose_dim) 
    x = init_x
    poses = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            data['sampled_pose'] = x
            data['t'] = batch_time_step
            grad = score_model(data)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  

            # normalisation
            if pose_mode == 'quat_wxyz' or pose_mode == 'quat_xyzw':
                # quat, should be normalised
                x[:, :4] /= torch.norm(x[:, :4], dim=-1, keepdim=True)   
            elif pose_mode == 'euler_xyz':
                pass
            else:
                # rotation(x axis, y axis), should be normalised
                x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
                x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)

            # Predictor step (Euler-Maruyama)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2*grad # R-SDE
            mean_x = x + drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            
            # normalisation
            x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            poses.append(x.unsqueeze(0))
    
    xs = torch.cat(poses, dim=0)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    mean_x[:, -3:] += data['pts_center']
    mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    # The last step does not include any noise
    return xs.permute(1, 0, 2), mean_x 

from utils.Lie import inv_SO3, log_SO3, exp_so3, bracket_so3, inv_SE3, log_SE3, bracket_se3, exp_se3, large_adjoint, SE3_geodesic_dist
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth


import time
@torch.no_grad()
def cond_ode_sampler(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz', 
        denoise=True,
        init_x=None,
        prob_threshold=0.5,      # 保留前30%高概率样本[7](@ref)
        cluster_bandwidth=0.15,  # 姿态空间聚类粒度[4](@ref)
        min_samples=5            # 最小簇样本数
    ):
    ###############################################################################################
    guidance = 1.0
    #start_time = time.time()
    def guided_vector_field(data, t, x_t):
        data['t'] = t
        x_t = x_t.detach().requires_grad_(True)  # Enable gradients for x_t
        data['sampled_pose'] = x_t
        num_samples = 10
        with torch.enable_grad():  # Enable gradient computation
            v_t = guidance * score_model(data)
            # eps = torch.randn_like(v_t)
            # jvp = torch.autograd.grad(v_t, x_t, grad_outputs=eps, create_graph=False, retain_graph=True)[0]
            # trace_estimate = torch.sum(eps * jvp, dim=1)  # Per-sample trace

            # Estimate the score function s_t(x_t) using a variant of Hutchinson's method
            trace_estimate = torch.zeros(x_t.shape[0], device=x_t.device)
            for _ in range(num_samples):
                eps = torch.randn_like(x_t)
                # 计算JVP (雅可比向量积) J(x_t)·ε
                jvp = torch.autograd.grad(
                    outputs=v_t,
                    inputs=x_t,
                    grad_outputs=eps,
                    create_graph=False,
                    retain_graph=True
                )[0]
                #trace_estimate += jvp

                # 计算迹估计的一个样本: ε·(J·ε)
                trace_sample = torch.sum(eps * jvp, dim=1)
                trace_estimate += trace_sample
            # 取平均得到最终迹估计
            trace_estimate /= num_samples
        # print(trace_estimate.mean())
        # exit()
        return v_t, trace_estimate.detach()

    # def guided_vector_field(data, t, x_t):
    #     data['t'] = t
    #     x_t = x_t.detach().requires_grad_(True)  # Enable gradients for x_t
    #     data['sampled_pose'] = x_t
    #     num_samples = 20
    #     with torch.enable_grad():  # Enable gradient computation
    #         v_t = guidance * score_model(data)
            
    #         # 并行计算迹估计
    #         batch_size, *data_dims = x_t.shape
            
    #         # # 扩展 x_t 维度以并行处理多个随机向量样本
    #         # x_t_expanded = x_t.unsqueeze(1).expand(-1, num_samples, *data_dims)
    #         # x_t_expanded = x_t_expanded.reshape(batch_size * num_samples, *data_dims)
            
    #         # 扩展x_t维度
    #         x_t_expanded = x_t.unsqueeze(1).expand(-1, num_samples, *x_t.shape[1:])
    #         x_t_expanded = x_t_expanded.reshape(batch_size * num_samples, *x_t.shape[1:])
        
    #         # 智能扩展t，使其与x_t的batch维度一致
    #         # 确保t的第一维是batch_size，然后扩展
    #         t_expanded = t.expand(batch_size, *t.shape[1:])  # 确保batch维度正确
    #         t_expanded = t_expanded.unsqueeze(1).expand(-1, num_samples, *t_expanded.shape[1:])
    #         t_expanded = t_expanded.reshape(batch_size * num_samples, *t_expanded.shape[2:])
            
    #         # 扩展pts_feat
    #         pts_feat = data['pts_feat']
    #         pts_feat_expanded = pts_feat.unsqueeze(1).expand(-1, num_samples, *pts_feat.shape[1:])
    #         pts_feat_expanded = pts_feat_expanded.reshape(batch_size * num_samples, *pts_feat.shape[1:])
        
            
    #         # 构建扩展后的 data 字典
    #         expanded_data = {
    #             't': t_expanded,
    #             'sampled_pose': x_t_expanded,
    #             'pts_feat': pts_feat_expanded
    #         }
            
    #         # 保留 data 中的其他字段（不扩展）
    #         # for key, value in data.items():
    #         #     if key not in ['t', 'sampled_pose', 'pts_feat']:
    #         #         expanded_data[key] = value
            
    #         # 为每个样本创建多个随机向量
    #         eps = torch.randn_like(x_t_expanded)
            
    #         # 计算 JVP (雅可比向量积) J(x_t)·ε
    #         v_t_expanded = guidance * score_model(expanded_data)
            
    #         jvp = torch.autograd.grad(
    #             outputs=v_t_expanded,
    #             inputs=x_t_expanded,
    #             grad_outputs=eps,
    #             create_graph=False,
    #             retain_graph=True
    #         )[0]
            
    #         # 计算迹估计: ε·(J·ε)
    #         trace_samples = (eps * jvp).reshape(batch_size, num_samples, -1).sum(dim=2)
            
    #         # 取平均得到最终迹估计
    #         trace_estimate = trace_samples.mean(dim=1)
            
    #     return v_t, trace_estimate.detach()

    repeat_num = 50
    bs = int(data['pts'].shape[0]/repeat_num)

    ode_solver = get_ode_solver('SE3_RK_mk', 20)

    # Sample initial x_0 and compute prior log probability
    x_0, log_prior = SO3_uniform_R3_normal(data['pts'].shape[0], data['pts'].device)
    # Log prior: R^3 components from normal distribution, SO(3) uniform (constant)
    #x_0_trans = x_0[:, :3, 3]
    #x_0_trans = x_0[:, :3, :]
    #log_prior = -0.5 * (x_0_trans**2).sum(dim=1) - 0.5 * np.log(2*np.pi)*6
    log_prior = log_prior.to(x_0.device)

    # Solve ODE and compute log probability
    traj, log_prob_integral = ode_solver(data, x_0, guided_vector_field)
    x_1_hat = traj[:, -1]  # Get final positions

    # Post-processing for poses
    x_1_hat_r = x_1_hat[:, :3, :3]
    trans_t = x_1_hat[:, :3, 3] + data['pts_center']
    #x_1_hat_r = x_1_hat_r.permute(0, 2, 1)
    x_axis = x_1_hat_r[:, 0, :3]
    y_axis = x_1_hat_r[:, 1, :3]
    #z_axis = x_1_hat_r[:, 2, :3]
    x_1_hat = torch.cat([x_axis, y_axis, trans_t], dim=1)
    # x_1_hat[:, :-3] = normalize_rotation(x_1_hat[:, :-3], 'rot_matrix')
    # x_1_hat[:, -3:] += data['pts_center']

    x_1_hat = x_1_hat.reshape(bs, repeat_num, -1)


    # def estimate_bandwidth_torch(points, quantile=0.3, n_samples=500):
    #     """PyTorch实现的带宽估计函数，通过分位数计算核窗口大小"""
    #     if points.size(0) > n_samples:
    #         idx = torch.randperm(points.size(0))[:n_samples]
    #         points = points[idx]
    #     # 计算成对距离矩阵的优化实现
    #     distances = torch.cdist(points, points)
    #     triu_mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
    #     sampled_distances = distances[triu_mask]
    #     return torch.quantile(sampled_distances, quantile)

    # def mean_shift_torch(points, bandwidth, max_iter=50, tol=1e-4):
    #     """完全基于PyTorch的MeanShift聚类核心算法"""
    #     device = points.device
    #     shift_points = points.clone()
    #     cluster_centers = []
        
    #     with torch.no_grad():
    #         for _ in range(max_iter):
    #             dists = torch.cdist(shift_points, shift_points)
    #             weights = torch.exp(-0.5 * (dists / bandwidth)**2)
    #             weights_sum = weights.sum(dim=1, keepdim=True)
    #             shift_vectors = (weights @ shift_points) / weights_sum
    #             shift_diffs = torch.norm(shift_vectors - shift_points, dim=1)
    #             shift_points = shift_vectors
    #             if torch.max(shift_diffs) < tol:
    #                 break

    #         # 合并邻近聚类中心
    #         while len(shift_points) > 0:
    #             center = shift_points[0:1]
    #             dists = torch.norm(shift_points - center, dim=1)
    #             cluster_mask = dists < bandwidth
    #             cluster_centers.append(shift_points[cluster_mask].mean(dim=0))
    #             shift_points = shift_points[~cluster_mask]
        
    #     return torch.stack(cluster_centers) if cluster_centers else torch.zeros(0, device=device)

    # #bs = data['pts'].size(0)
    # device = data['pts'].device
    
    # #x_1_hat = data['x_1_hat']
    # log_prior = log_prior.reshape(bs, repeat_num)
    # log_prob_integral = log_prob_integral.reshape(bs, repeat_num)
    
    # log_prob = (log_prior - log_prob_integral).to(device)
    # prob = torch.exp(log_prob - log_prob.max(dim=1, keepdim=True).values)
    
    # final_poses_list = []
    # for batch_idx in range(bs):
    #     current_x_hat = x_1_hat[batch_idx]  # [50, 6]
    #     current_prob = prob[batch_idx]  # [50]
        
    #     # 有效点选择优化
    #     valid_mask = current_prob > torch.quantile(current_prob, 0.25)
    #     if valid_mask.sum() < 25:
    #         topk_idx = torch.topk(current_prob, 25).indices
    #         valid_mask = torch.zeros_like(current_prob, dtype=torch.bool)
    #         valid_mask[topk_idx] = True
        
    #     valid_points = current_x_hat[valid_mask]
    #     if valid_points.size(0) == 0:
    #         final_poses_list.append(torch.zeros(0, 9, device=device))
    #         continue
            
    #     # 带宽估计优化
    #     bandwidth = estimate_bandwidth_torch(valid_points, quantile=0.3)
    #     if bandwidth < 1e-6:
    #         final_poses_list.append(valid_points.mean(dim=0, keepdim=True))
    #         continue
            
    #     # GPU加速的MeanShift
    #     cluster_centers = mean_shift_torch(valid_points, bandwidth)
        
    #     # 聚类概率计算优化
    #     if cluster_centers.size(0) > 0:
    #         dists = torch.cdist(valid_points, cluster_centers)
    #         labels = torch.argmin(dists, dim=1)
    #         cluster_probs = torch.stack([current_prob[valid_mask][labels == i].mean() 
    #                                   for i in range(cluster_centers.size(0))])
    #         top_cluster = cluster_centers[torch.argmax(cluster_probs)].unsqueeze(0)
    #         # 旋转矩阵归一化
    #         #top_cluster[:, :-3] = F.normalize(top_cluster[:, :-3], dim=1)
    #         top_cluster[:, :-3] = normalize_rotation(top_cluster[:, :-3], 'rot_matrix')
    #         final_poses_list.append(top_cluster)




    # 多batch处理（移除numpy转换）
    x_1_hat_tensor = x_1_hat  # 直接使用tensor
    log_prior = log_prior.reshape(bs, repeat_num)
    log_prob_integral = log_prob_integral.reshape(bs, repeat_num)
    log_prob = (log_prior - log_prob_integral)  # 保持tensor格式

    # final_poses_list = []
    # for batch_idx in range(bs):
    #     current_x_hat = x_1_hat_tensor[batch_idx]
    #     current_log_prob = log_prob[batch_idx]

    #     # 数值稳定性处理（使用PyTorch操作）
    #     prob = torch.exp(current_log_prob - current_log_prob.max())  # [4](@ref)

    #     top_values, valid_mask = torch.topk(prob, k=30, largest=True, sorted=True)

    #     # 提取有效点及其概率（保持tensor）
    #     valid_points = current_x_hat[valid_mask]
    #     valid_probs = prob[valid_mask]

    #     final_poses = weighted_fusion_poses(valid_points, valid_probs).unsqueeze(0)
    #     final_poses[:, :-3] = normalize_rotation(final_poses[:, :-3], 'rot_matrix')
    #     final_poses_list.append(final_poses)

    # 数值稳定性处理（使用PyTorch操作）
    prob = torch.exp(log_prob - log_prob.max(dim=1, keepdim=True)[0])

    # 筛选有效点（使用torch.quantile）
    # 这里使用topk来选择前30个最大的概率对应的点
    top_values, top_indices = torch.topk(prob, k=20, largest=True, sorted=True, dim=1)

    # 提取有效点及其概率（保持tensor）
    valid_points = torch.gather(x_1_hat_tensor, 1, top_indices.unsqueeze(-1).expand(-1, -1, 9))
    valid_probs = torch.gather(prob, 1, top_indices)

    #final_poses = weighted_fusion_poses(valid_points, valid_probs)
    #final_poses[:, :-3] = normalize_rotation(final_poses[:, :-3], 'rot_matrix')

    rot_matrix = get_rot_matrix(valid_points[:, :, :-3].reshape(bs * 20, -1), 'rot_matrix')
    quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, 20, -1)
    aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz, weights=valid_probs)


    if True:
        for j in range(bs):
            # https://math.stackexchange.com/a/90098
            # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
            pairwise_distance = 1 - torch.sum(quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1), dim=2) ** 2
            dbscan = DBSCAN(eps=0.05, min_samples=int(0.1667 * 20)).fit(pairwise_distance.cpu().cpu().numpy())
            labels = dbscan.labels_
            if np.any(labels >= 0):
                bins = np.bincount(labels[labels >= 0])
                best_label = np.argmax(bins)
                aggregated_quat_wxyz[j] = average_quaternion_batch(quat_wxyz[j, labels == best_label].unsqueeze(0), \
                    weights=valid_probs[j, labels == best_label].unsqueeze(0))[0]


    aggregated_trans = torch.mean(valid_points[:, :, -3:], dim=1)
    aggregated_pose = torch.zeros(bs, 4, 4)
    aggregated_pose[:, 3, 3] = 1
    aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz).permute(0, 2, 1)
    aggregated_pose[:, :3, 3] = aggregated_trans

    rot = aggregated_pose[:, :2, :3].flatten(1, 2)
    trans = aggregated_pose[:, :3, 3]
    aggregated_pose = torch.cat([rot, trans], dim=1)

    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"代码执行时间为: {execution_time} 秒")
    # exit()

    # print('aggregated_quat_wxyz: ', aggregated_quat_wxyz.shape)
    # exit()






    # # 多batch处理
    # x_1_hat_np = x_1_hat.cpu().numpy()
    # log_prior = log_prior.reshape(bs, repeat_num)
    # log_prob_integral = log_prob_integral.reshape(bs, repeat_num)
    # log_prob = (log_prior - log_prob_integral).cpu().numpy()

    # final_poses_list = []
    # for batch_idx in range(bs):
    #     current_x_hat = x_1_hat_np[batch_idx]
    #     current_log_prob = log_prob[batch_idx]

    #     prob = np.exp(current_log_prob - current_log_prob.max())
    #     # print(prob)
    #     # exit()
    #     #valid_mask = prob > np.quantile(prob, prob_threshold)
    #     # 修改后：降低阈值至0.3，保留更多点
        
    #     valid_mask = prob > np.quantile(prob, 0.5)  # 进一步放宽阈值至1%
    #     if np.sum(valid_mask) < 10:  # 若数据过少，启用备用选择
    #         top_k_indices = np.argpartition(-prob, 10)[:10]
    #         valid_mask = np.zeros_like(prob, dtype=bool)
    #         valid_mask[top_k_indices] = True
    #     print(np.sum(valid_mask))

    #     bandwidth = estimate_bandwidth(current_x_hat[valid_mask], quantile=0.3, n_samples=500)
    #     #bandwidth = estimate_bandwidth(current_x_hat[valid_mask], quantile=0.1, n_samples=500) * 1.2

    #     #print(bandwidth)
    #     # cluster = MeanShift(
    #     #     bandwidth=bandwidth,
    #     #     min_bin_freq=1,
    #     #     bin_seeding=True
    #     # ).fit(current_x_hat[valid_mask])
    #     cluster = MeanShift(
    #         bandwidth=bandwidth,
    #         bin_seeding=True,
    #         min_bin_freq=1,  # 确保至少一个分箱有数据
    #         cluster_all=True
    #     ).fit(current_x_hat[valid_mask])

    #     if len(cluster.cluster_centers_) > 0:
    #         cluster_probs = [
    #             prob[valid_mask][cluster.labels_ == i].mean()
    #             for i in np.unique(cluster.labels_)
    #         ]
    #         #print(len(cluster_probs))
    #         top_clusters = np.argsort(cluster_probs)[-1:][::-1]
    #         #print(len(top_clusters))
    #         final_poses = cluster.cluster_centers_[top_clusters]
    #     else:
    #         final_poses = np.array([])

    #     if final_poses.size > 0:
    #         final_poses_tensor = torch.tensor(final_poses).to(data['pts'].device)
    #         final_poses_tensor[:, :-3] = normalize_rotation(final_poses_tensor[:, :-3], 'rot_matrix')
    #         #final_poses_tensor[:, -3:] /= 2.0
    #         final_poses_list.append(final_poses_tensor)


    # agg_poses = torch.cat(final_poses_list, dim=0)

    # print(aggregated_pose[0])
    # print(aggregated_pose[1])
    # print(aggregated_pose[2])
    # print(aggregated_pose[3])
    # print(aggregated_pose[4])
    # exit()
    
    final_poses = aggregated_pose.unsqueeze(1).repeat(1, repeat_num, 1).flatten(0, 1)

    # final_poses = x_1_hat_tensor.flatten(0, 1)
    # final_prob = prob.flatten(0, 1)

    return None, final_poses




def cond_edm_sampler(
    decoder_model, data, prior_fn, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    pose_mode='quat_wxyz', device='cuda'
):
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    latents = prior_fn((batch_size, pose_dim)).to(device)

    # Time step discretization. note that sigma and t is interchangable
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    def decoder_wrapper(decoder, data, x, t):
        # save temp
        x_, t_= data['sampled_pose'], data['t']
        # init data
        data['sampled_pose'], data['t'] = x, t
        # denoise
        data, denoised = decoder(data)
        # recover data
        data['sampled_pose'], data['t'] = x_, t_
        return denoised.to(torch.float64)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    xs = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = decoder_wrapper(decoder_model, data, x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = decoder_wrapper(decoder_model, data, x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        xs.append(x_next.unsqueeze(0))

    xs = torch.stack(xs, dim=0) # [num_steps, bs, pose_dim]
    x = xs[-1] # [bs, pose_dim]

    # post-processing
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x


