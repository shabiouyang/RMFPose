import argparse
from ipdb import set_trace

def get_config():
    parser = argparse.ArgumentParser()
    
    """ dataset """
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--max_batch_size', type=int, default=192)
    parser.add_argument('--pose_mode', type=str, default='rot_matrix')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--percentage_data_for_train', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_val', type=float, default=1.0) 
    parser.add_argument('--percentage_data_for_test', type=float, default=1.0) 
    parser.add_argument('--train_source', type=str, default='Omni6DPose')
    parser.add_argument('--val_source', type=str, default='scannet++')
    parser.add_argument('--test_source', type=str, default='scannet++')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--per_obj', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--omni6dmug', action='store_true') # set to true if evaluating/fintuning an omni6dpose model on NOCS data
    
    
    """ model """
    parser.add_argument('--sampler_mode', nargs='+')
    parser.add_argument('--sampling_steps', type=int)
    parser.add_argument('--sde_mode', type=str, default='ve')
    parser.add_argument('--regression_head', type=str, default='Rx_Ry_and_T')
    parser.add_argument('--pointnet2_params', type=str, default='light')
    parser.add_argument('--pts_encoder', type=str, default='pointnet2') 
    parser.add_argument('--energy_mode', type=str, default='IP') 
    parser.add_argument('--s_theta_mode', type=str, default='score') 
    parser.add_argument('--norm_energy', type=str, default='identical')
    parser.add_argument('--dino', type=str, default='pointwise') # none / global / pointwise
    parser.add_argument('--scale_embedding', type=int, default=180)
    
    
    """ training """
    parser.add_argument('--agent_type', type=str, default='score', help='one of the [score, energy, energy_with_ranking, scale]')
    parser.add_argument('--pretrained_score_model_path', type=str)
    parser.add_argument('--pretrained_energy_model_path', type=str)
    parser.add_argument('--pretrained_scale_model_path', type=str)
    parser.add_argument('--distillation', default=False, action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--optimizer',  type=str, default='Adam')
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--use_pretrain', default=False, action='store_true')
    parser.add_argument('--parallel', default=False, action='store_true')   
    parser.add_argument('--num_gpu', type=int, default=4)
    parser.add_argument('--is_train', default=False, action='store_true')
    parser.add_argument('--perfect_depth', default=False, action='store_true')
    parser.add_argument('--load_per_object', default=False, action='store_true')
    parser.add_argument('--scale_batch_size', type=int, default=64)
    
    
    """ testing """
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--pred', default=False, action='store_true')
    parser.add_argument('--eval_repeat_num', type=int, default=5)
    parser.add_argument('--real_drop', type=int, default=10) # only keep part of the frames in test set for faster inference
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--T0', type=float, default=1.0)
    
    
    parser.add_argument('--img_size', type=int, default=224, help='cropped image size')
    parser.add_argument('--result_dir', type=str, default='', help='result directory')
    parser.add_argument('--clustering', type=int, default=1, help='use clustering to solve multimodal issue')
    parser.add_argument('--clustering_eps', type=float, default=0.05, 
                        help='hyperparameter in clustering (see runners/evaluation_single.py for details)')
    parser.add_argument('--clustering_minpts', type=float, default=0.1667, 
                        help='hyperparameter in clustering (see runners/evaluation_single.py for details)')
    parser.add_argument('--retain_ratio', type=float, default=0.4, help='how much to retain in outlier removal stage')
    
    
    cfg = parser.parse_args()
    
    # some of these augmentation parameters are only for NOCS training
    # dynamic zoom in parameters
    cfg.DYNAMIC_ZOOM_IN_PARAMS = {
        'DZI_PAD_SCALE': 1.5,
        'DZI_TYPE': 'uniform',
        'DZI_SCALE_RATIO': 0.25,
        'DZI_SHIFT_RATIO': 0.25
    }
    # pts aug parameters
    cfg.PTS_AUG_PARAMS = {
        'aug_pc_pro': 0.2,
        'aug_pc_r': 0.2,
        'aug_rt_pro': 0.3,
        'aug_bb_pro': 0.3,
        'aug_bc_pro': 0.3
    }
    # 2D aug parameters
    cfg.DEFORM_2D_PARAMS = {
        'roi_mask_r': 3,
        'roi_mask_pro': 0.5
    }
    if cfg.eval or cfg.pred: # disable augmentation in evaluation
        cfg.DYNAMIC_ZOOM_IN_PARAMS['DZI_TYPE'] = 'none'
        cfg.DEFORM_2D_PARAMS['roi_mask_pro'] = 0

    assert cfg.dino in ['none', 'global', 'pointwise']
    
    return cfg

# @torch.no_grad()
# def cond_ode_sampler(
#         score_model,
#         data,
#         prior,
#         sde_coeff,
#         atol=1e-5, 
#         rtol=1e-5, 
#         device='cuda', 
#         eps=1e-5,
#         T=1.0,
#         num_steps=None,
#         pose_mode='quat_wxyz', 
#         denoise=True,
#         init_x=None,
#     ):
#     ###############################################################################################
#     guidance = 1.0
#     def guided_vector_field(data, t, x_t):
#         data['t'] = t
#         data['sampled_pose'] = x_t
#         with torch.no_grad():
#             v_t = guidance * score_model(data)
#         return v_t


#     repeat_num = 50
#     bs = int(data['pts'].shape[0]/repeat_num)

#     ode_solver = get_ode_solver('SE3_RK_mk', 100)

#     x_0 = SO3_uniform_R3_normal(data['pts'].shape[0], data['pts'].device)

#     x_1_hat = ode_solver(data, x_0, guided_vector_field)[:, -1]

#     x_1_hat = x_1_hat.permute(0, 2, 1)
#     x_axis = x_1_hat[:, 0, :3]  # [bs,3]  旋转矩阵的第一行（x轴方向）
#     y_axis = x_1_hat[:, 1, :3]  # [bs,3]  旋转矩阵的第二行（y轴方向）
#     #z_axis = x_1_hat[:, 2, :3]  # [bs,3]  旋转矩阵的第三行（z轴方向）
#     trans_t = x_1_hat[:, :3, 3]
#     x_1_hat = torch.cat([
#         x_axis,      # x轴方向分量 [bs,3]
#         y_axis,      # y轴方向分量 [bs,3]
#         #z_axis,      # z轴方向分量 [bs,3]
#         trans_t      # 平移分量    [bs,3]
#     ], dim=1)        # 最终形状 [bs, 3+3+3=9]

#     x_1_hat[:, :-3] = normalize_rotation(x_1_hat[:, :-3], 'rot_matrix')
#     x_1_hat[:, -3:] += data['pts_center']

#     x_1_hat = x_1_hat.cpu().numpy()

#     return None, x_1_hat
# class SE3_RK4_MK:
#     def __init__(self, num_steps):
#         self.t = torch.linspace(0, 1, num_steps + 1)

#     @torch.no_grad()
#     def __call__(self, data, x_0, func):
#         # Initialize
#         t = self.t.to(x_0.device)
#         dt = t[1:] - t[:-1]
#         traj = x_0.new_zeros(x_0.shape[0:1] + t.shape + x_0.shape[1:])
#         traj[:, 0] = x_0

#         # print('traj: ', traj.shape)
#         # print('x_0: ', x_0.shape)

#         for n in range(len(t)-1):
#             # Get n-th values
#             x_n = traj[:, n].contiguous()
#             t_n = t[n].repeat(len(x_0), 1)
#             h = dt[n].repeat(len(x_0), 1)

#             ##### Stage 1 #####
#             # Set function input
#             x_hat_1 = x_n

#             # Get vector (V_s)
#             V_1 = func(data, t_n, bracket_se3(log_SE3(x_hat_1)))
#             w_1 = V_1[:, :3]
#             v_1 = V_1[:, 3:]

#             # print('x_hat_1: ', x_hat_1.shape)
#             # print('w_1: ', w_1.shape)
#             # exit()

#             # Change w_s to w_b and transform to matrix
#             w_1 = torch.einsum('bji,bj->bi', x_hat_1[:, :3, :3], w_1)
#             w_1 = bracket_so3(w_1)

#             # Set I_1
#             I_1 = w_1

#             ##### Stage 2 #####
#             u_2 = h.unsqueeze(-1) * (1 / 2) * w_1
#             u_2 += (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_2)

#             # Set function input
#             x_hat_2 = deepcopy(x_n)
#             x_hat_2[:, :3, :3] @= exp_so3(u_2)
#             x_hat_2[:, :3, 3] += h * (v_1 / 2)

#             # Get vector (V_s)
#             V_2 = func(data, t_n + (h / 2), bracket_se3(log_SE3(x_hat_2)))
#             w_2 = V_2[:, :3]
#             v_2 = V_2[:, 3:]

#             # Change w_s to w_b and transform to matrix
#             w_2 = torch.einsum('bji,bj->bi', x_hat_2[:, :3, :3], w_2)
#             w_2 = bracket_so3(w_2)

#             ##### Stage 3 #####
#             u_3 = h.unsqueeze(-1) * (1 / 2) * w_2
#             u_3 += (h.unsqueeze(-1) / 12) * Lie_bracket(I_1, u_3)

#             # Set function input
#             x_hat_3 = deepcopy(x_n)
#             x_hat_3[:, :3, :3] @= exp_so3(u_3)
#             x_hat_3[:, :3, 3] += h * (v_2 / 2)

#             # Get vector (V_s)
#             V_3 = func(data, t_n + (h / 2), bracket_se3(log_SE3(x_hat_3)))
#             w_3 = V_3[:, :3]
#             v_3 = V_3[:, 3:]

#             # Change w_s to w_b and transform to matrix
#             w_3 = torch.einsum('bji,bj->bi', x_hat_3[:, :3, :3], w_3)
#             w_3 = bracket_so3(w_3)

#             ##### Stage 4 #####
#             u_4 = h.unsqueeze(-1) * w_3
#             u_4 += (h.unsqueeze(-1) / 6) * Lie_bracket(I_1, u_4)

#             # Set function input
#             x_hat_4 = deepcopy(x_n)
#             x_hat_4[:, :3, :3] @= exp_so3(u_4)
#             x_hat_4[:, :3, 3] += h * v_3

#             # Get vector (V_s)
#             V_4 = func(data, t_n + h, bracket_se3(log_SE3(x_hat_4)))
#             w_4 = V_4[:, :3]
#             v_4 = V_4[:, 3:]

#             # Change w_s to w_b and transform to matrix
#             w_4 = torch.einsum('bji,bj->bi', x_hat_4[:, :3, :3], w_4)
#             w_4 = bracket_so3(w_4)

#             ##### Update #####
#             I_2 = (2 * (w_2 - I_1) + 2 * (w_3 - I_1) - (w_4 - I_1)) / h.unsqueeze(-1)
#             u = h.unsqueeze(-1) * (1 / 6 * w_1 + 1 / 3 * w_2 + 1 / 3 * w_3 + 1 / 6 * w_4)
#             u += (h.unsqueeze(-1) / 4) * Lie_bracket(I_1, u) + ((h ** 2).unsqueeze(-1) / 24) * Lie_bracket(I_2, u)

#             traj[:, n+1] = deepcopy(x_n)
#             traj[:, n+1, :3, :3] @= exp_so3(u)
#             traj[:, n+1, :3, 3] += (h / 6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4)

#         return traj