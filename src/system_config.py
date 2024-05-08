from typing import NamedTuple
from typing import Any
import torch
from math import pi

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class SystemConfig:
    @staticmethod
    def get_config(system_name: str):
        if system_name == 'dubins': return Dubins
        elif system_name == 'reedsshepp': return ReedsShepp
        elif system_name == 'quadrotor': return Quadrotor
        elif system_name == 'acrobot': return Acrobot
        elif system_name == 'cartpole': return Cartpole
        elif system_name == 'nimblearm': return NimbleArmCartesian
        elif system_name == 'nimblearmjoint': return NimbleArmJoint
        elif system_name == 'nimblehusky': return NimbleHusky
        # elif system_name == 'single_integrator': return f_singleintegrator
        else: raise NotImplementedError()


class Dubins(NamedTuple):
    x_star: float = 0.
    u_star: float = 0.


class ReedsShepp(NamedTuple):
    x_star: float = 0.
    u_star: float = 0.

    w_P: float = 1.
    w_Q: float = 0.
    w_R: float = 0.

    init_dist: torch.distributions.Uniform = torch.distributions.Uniform(torch.Tensor([-3.5, -3, -pi]), torch.Tensor([-2.5, 3, pi]))

    x_lims: list = [-3, 3]
    output_scaling: torch.tensor = torch.Tensor([-1, 1]).to(device) # controller limits
    x_dim: int = 3
    u_dim: int = 2

    fk_out_scale: float = 1.
    fk_scale_dims: Any = None

    obstacle_centers: torch.tensor = torch.tensor([[-1., 0.], [1., 0.]]).to(device)


class Quadrotor(NamedTuple):
    x_star: torch.tensor = torch.Tensor([3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    u_star: float = 0.
    w_P: torch.tensor = torch.Tensor([1., 1., 1., 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    w_Q: float = 0.
    w_R: float = 0.

    init_dist: torch.distributions.normal.Normal = torch.distributions.Normal(torch.zeros(12), 1 * torch.ones(12))

    x_lims: list = [-5, 5]
    output_scaling: torch.tensor = torch.Tensor([-100, 100]).to(device) # controller limits
    x_dim: int = 12
    u_dim: int = 4

    fk_out_scale: float = 100.
    fk_scale_dims: list = torch.arange(6, 12).tolist()

    obstacle_centers: torch.tensor = torch.tensor([[ 1.5, 1.5, 2.5]]).to(device)


class Cartpole(NamedTuple):
    x_star: torch.tensor = torch.Tensor([0, 0, 0, 0]).to(device)
    u_star: float = 0.
    w_P: torch.tensor = torch.Tensor([1., 0, 1., 0]).to(device)
    w_Q: float = 0.
    w_R: float = 0.

    init_dist: torch.distributions.normal.Normal = torch.distributions.Normal(torch.Tensor([0., 0., pi, 0.]), 0.1 * torch.ones(4))

    x_lims: list = [-5, 5]
    output_scaling: torch.tensor = torch.Tensor([-3, 3]).to(device) # controller limits
    x_dim: int = 4
    u_dim: int = 2

    fk_out_scale: float = 1.
    fk_scale_dims: Any = None


class Acrobot(NamedTuple):
    x_star: torch.tensor = torch.Tensor([pi, 0, 0, 0]).to(device)
    u_star: float = 0.
    w_P: torch.tensor = torch.Tensor([1., 1., 0, 0]).to(device)
    w_Q: float = 0.
    w_R: float = 0.

    init_dist: torch.distributions.normal.Normal = torch.distributions.Normal(torch.zeros(4), 0.1 * torch.ones(4))

    x_lims: list = [-6, 6]
    output_scaling: torch.tensor = torch.Tensor([-5, 5]).to(device) # controller limits
    x_dim: int = 4
    u_dim: int = 2

    fk_out_scale: float = 1.
    fk_scale_dims: Any = None


class NimbleArmCartesian(NamedTuple):
    x_star: torch.tensor = torch.Tensor([0., 0.8, -1.0]).to(device)
    u_star: float = 0.
    w_P: float = 1.
    w_Q: float = 0.
    w_R: float = 0.

    init_dist: torch.distributions.normal.Normal = torch.distributions.Normal(torch.Tensor([0.42, 0.193, 0.0006]), 0.5 * torch.ones(3))

    x_lims: list = [-5, 5]
    output_scaling: torch.tensor = torch.Tensor([-10, 10]).to(device) # controller limits
    x_dim: int = 3
    u_dim: int = 6

    fk_out_scale: float = 1.
    fk_scale_dims: Any = None


class NimbleArmJoint(NamedTuple):

    x_lims: list = [-5, 5]
    output_scaling: torch.tensor = torch.Tensor([-10, 10]).to(device) # controller limits
    x_dim: int = 12
    u_dim: int = 6
    
    x_star: torch.tensor = torch.tensor([-1.3256,  1.4205, -0.4190,  0.3577, -0.1508,  0.1087, -0.1135, -1.3955,
        -1.3766, -0.0968, -0.9656,  0.2583]).to(device)
    u_star: float = 0.
    w_P: float = 1.
    w_Q: float = 0.
    w_R: float = 0.

    init_dist: torch.distributions.normal.Normal = torch.distributions.Normal(torch.zeros(x_dim), 0.1 * torch.ones(x_dim))

    fk_out_scale: float = 1.
    fk_scale_dims: Any = None


class NimbleHusky(NamedTuple):
    output_scaling: torch.tensor = torch.Tensor([-30, 30]).to(device) # controller limits
    x_dim: int = 20
    u_dim: int = 2

    state_lb = torch.zeros(x_dim)
    state_ub = torch.zeros(x_dim)

    # state_lb[:3] = -pi / 2
    # state_ub[:3] = 3 * pi / 2

    state_lb[:3] = -0.001
    state_ub[:3] = 0.001

    state_lb[3] = -0.005
    state_lb[5] = -1.
    state_ub[3] = 0.005
    state_ub[5] = 1.
    height = -0.0402
    state_lb[4] = height - 0.0002
    state_ub[4] = height + 0.0002
    state_lb[6:10] = -0.01
    state_ub[6:10] = 0.01

    state_ub[10:] = 0.0001
    init_dist = torch.distributions.Uniform(state_lb, state_ub)

    x_star: torch.tensor = torch.cat((torch.tensor([0., 0., 0., 3., height, 0.]), torch.zeros(14))).to(device)
    u_star: float = 0.

    w_P: torch.tensor = torch.cat((torch.tensor([0., 0., 0., 100., 0., 100.]), torch.zeros(14))).to(device)
    # w_P: torch.tensor = torch.cat((torch.ones(x_dim // 2), torch.zeros(x_dim // 2))) .to(device)
    w_Q: float = 0.
    w_R: float = 0.


    fk_out_scale: float = 1.
    fk_scale_dims: Any = None


class NimbleRover(NamedTuple):
    output_scaling: torch.tensor = torch.Tensor([-5, 5]).to(device) # controller limits
    x_dim: int = 20
    u_dim: int = 2

    x_temp = torch.zeros(x_dim)
    x_temp[:3] = -pi / 2
    x_temp[4] = 0.304

    x_temp[3] = 0.
    x_temp[5] = 5.

    x_star: torch.tensor = x_temp.to(device)
    u_star: float = 0.

    w_P: torch.tensor = (x_temp / 1.).to(device)
    w_Q: float = 0.
    w_R: float = 0.

    mean_ = torch.zeros(x_dim)
    mean_[:3] = -pi / 2
    mean_[4] = 0.304

    cov_ = torch.ones(x_dim)
    cov_[4] = 0.001
    cov_[10:] = 0.01

    init_dist: torch.distributions.normal.Normal = torch.distributions.Normal(mean_, cov_)

    fk_out_scale: float = 1.
    fk_scale_dims: Any = None
