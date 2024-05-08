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
