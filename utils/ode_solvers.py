import torch
from copy import deepcopy

from utils.Lie import Lie_bracket, inv_SO3, log_SO3, exp_so3, bracket_so3, inv_SE3, log_SE3, bracket_se3, exp_se3, large_adjoint, SE3_geodesic_dist


def se3_bracket(xi1, xi2):
    """SE(3)李括号运算[3](@ref)"""
    w1, v1 = xi1[:, :3], xi1[:, 3:]
    w2, v2 = xi2[:, :3], xi2[:, 3:]
    return torch.cat([
        torch.cross(w1, w2, dim=1),
        torch.cross(w1, v2, dim=1) + torch.cross(v1, w2, dim=1)
    ], dim=1)

def get_ode_solver(cfg, num_steps):

    if cfg == 'SE3_Euler':
        solver = SE3_Euler(num_steps)
    elif cfg == 'SE3_RK_mk':
        solver = SE3_RK4_MK(num_steps)
    else:
        raise NotImplementedError(f"ODE solver {name} is not implemented.")

    return solver

class SE3_Euler:
    def __init__(self, num_steps):
        self.t = torch.linspace(0, 1, num_steps + 1)

    @torch.no_grad()
    def __call__(self, data, x_0, func):
        # Initialize
        t = self.t.to(x_0.device)
        dt = t[1:] - t[:-1]
        traj = x_0.new_zeros(x_0.shape[0:1] + t.shape + x_0.shape[1:])
        traj[:, 0] = x_0
        log_prob_integral = torch.zeros(x_0.size(0), device=x_0.device)

        for n in range(len(t)-1):
            # Get n-th values
            x_n = traj[:, n].contiguous()
            t_n = t[n].repeat(x_0.size(0), 1)
            h = dt[n].repeat(x_0.size(0), 1)

            ##### Stage 1 #####
            # Set function input
            x_hat = deepcopy(x_n)

            # Get vector (V_s)
            V_1, trace = func(data, t_n, bracket_se3(log_SE3(x_hat)))
            w_1 = V_1[:, :3]
            v_1 = V_1[:, 3:]

            # ##### Stage 1 #####
            # x_hat_1 = x_n
            # V_1, trace_1 = func(data, t_n, bracket_se3(log_SE3(x_hat_1)))
            # w_1 = V_1[:, :3]
            # v_1 = V_1[:, 3:]
            # w_1 = torch.einsum('bji,bj->bi', x_hat_1[:, :3, :3], w_1)
            # w_1 = bracket_so3(w_1)
            # I_1 = w_1

            # Change w_s to w_b and transform to matrix
            w_1 = torch.einsum('bji,bj->bi', x_hat[:, :3, :3], w_1)
            w_1 = bracket_so3(w_1)

            trace_increment = trace*(h.squeeze(-1)/6)

            log_prob_integral += trace_increment

            ##### Update #####
            traj[:, n+1] = deepcopy(x_n)
            traj[:, n+1, :3, :3] @= exp_so3(h.unsqueeze(-1) * w_1)
            traj[:, n+1, :3, 3] += h * v_1

        return traj, log_prob_integral


class SE3_RK4_MK:
    def __init__(self, num_steps):
        self.t = torch.linspace(0, 1, num_steps + 1)

    @torch.no_grad()
    def __call__(self, data, x_0, func):
        t = self.t.to(x_0.device)
        dt = t[1:] - t[:-1]
        traj = x_0.new_zeros(x_0.shape[0:1] + t.shape + x_0.shape[1:])
        traj[:, 0] = x_0
        log_prob_integral = torch.zeros(x_0.size(0), device=x_0.device)

        for n in range(len(t)-1):
            x_n = traj[:, n].contiguous()
            t_n = t[n].repeat(x_0.size(0), 1)
            h = dt[n].repeat(x_0.size(0), 1)

            ##### Stage 1 #####
            x_hat_1 = x_n
            V_1, trace_1 = func(data, t_n, bracket_se3(log_SE3(x_hat_1)))
            w_1 = V_1[:, :3]
            v_1 = V_1[:, 3:]
            w_1 = torch.einsum('bji,bj->bi', x_hat_1[:, :3, :3], w_1)
            w_1 = bracket_so3(w_1)
            I_1 = w_1

            ##### Stage 2 #####
            # 分步计算避免未定义引用
            u_2_base = h.unsqueeze(-1) * (1/2) * w_1  # 基础项
            lie_bracket_term = (h.unsqueeze(-1)/12) * Lie_bracket(I_1, u_2_base)
            u_2 = u_2_base + lie_bracket_term

            x_hat_2 = deepcopy(x_n)
            x_hat_2[:, :3, :3] @= exp_so3(u_2)
            x_hat_2[:, :3, 3] += h * (v_1 / 2)
            V_2, trace_2 = func(data, t_n + (h / 2), bracket_se3(log_SE3(x_hat_2)))
            w_2 = V_2[:, :3]
            v_2 = V_2[:, 3:]
            w_2 = torch.einsum('bji,bj->bi', x_hat_2[:, :3, :3], w_2)
            w_2 = bracket_so3(w_2)

            ##### Stage 3 #####
            # 同样的分步计算
            u_3_base = h.unsqueeze(-1) * (1/2) * w_2
            lie_bracket_term = (h.unsqueeze(-1)/12) * Lie_bracket(I_1, u_3_base)
            u_3 = u_3_base + lie_bracket_term

            x_hat_3 = deepcopy(x_n)
            x_hat_3[:, :3, :3] @= exp_so3(u_3)
            x_hat_3[:, :3, 3] += h * (v_2 / 2)
            V_3, trace_3 = func(data, t_n + (h / 2), bracket_se3(log_SE3(x_hat_3)))
            w_3 = V_3[:, :3]
            v_3 = V_3[:, 3:]
            w_3 = torch.einsum('bji,bj->bi', x_hat_3[:, :3, :3], w_3)
            w_3 = bracket_so3(w_3)

            ##### Stage 4 #####
            u_4_base = h.unsqueeze(-1) * w_3
            lie_bracket_term = (h.unsqueeze(-1)/6) * Lie_bracket(I_1, u_4_base)
            u_4 = u_4_base + lie_bracket_term

            x_hat_4 = deepcopy(x_n)
            x_hat_4[:, :3, :3] @= exp_so3(u_4)
            x_hat_4[:, :3, 3] += h * v_3
            V_4, trace_4 = func(data, t_n + h, bracket_se3(log_SE3(x_hat_4)))
            w_4 = V_4[:, :3]
            v_4 = V_4[:, 3:]
            w_4 = torch.einsum('bji,bj->bi', x_hat_4[:, :3, :3], w_4)
            w_4 = bracket_so3(w_4)

            ##### Update #####
            # 计算迹增量（使用RK4系数）
            trace_increment = (trace_1 + 2*trace_2 + 2*trace_3 + trace_4) * (h.squeeze(-1)/6)
            log_prob_integral += trace_increment

            # 更新轨迹
            I_2 = (2 * (w_2 - I_1) + 2 * (w_3 - I_1) - (w_4 - I_1)) / h.unsqueeze(-1)
            u = h.unsqueeze(-1) * (1/6 * w_1 + 1/3 * w_2 + 1/3 * w_3 + 1/6 * w_4)
            u += (h.unsqueeze(-1)/4) * Lie_bracket(I_1, u) + ((h**2).unsqueeze(-1)/24) * Lie_bracket(I_2, u)
            
            traj[:, n+1] = deepcopy(x_n)
            traj[:, n+1, :3, :3] @= exp_so3(u)
            traj[:, n+1, :3, 3] += (h / 6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4)

        return traj, log_prob_integral