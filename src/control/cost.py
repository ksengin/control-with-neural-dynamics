import torch
import torch.nn as nn


class IntegralCost(nn.Module):
    '''Integral cost function
    Args:
        x_star: torch.tensor, target position
        u_star: torch.tensor / float, controller with no cost
        P: terminal cost weights
        Q: state weights
        R: controller regulator weights
    '''
    def __init__(self, x_star, u_star=0, P=0, Q=1, R=0):
        super().__init__()
        self.x_star = x_star
        self.u_star = u_star
        self.P, self.Q, self.R, = P, Q, R
        
    def forward(self, x, u=torch.Tensor([0.])):
        """
        x: trajectories
        u: control inputs
        """
        cost = torch.norm(self.P*(x[-1] - self.x_star), p=2, dim=-1).mean()
        cost += torch.norm(self.Q*(x - self.x_star), p=2, dim=-1).mean()
        cost += torch.norm(self.R*(u - self.u_star), p=2, dim=-1).mean()
        return cost
