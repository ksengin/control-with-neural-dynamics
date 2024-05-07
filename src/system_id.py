import argparse
import numpy as np
import time
import datetime
import torch
from torch import nn
from torch import optim
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm, trange
import json

from models.fknet import FKNet
from models.diff_ops import gradient, batch_jacobian
from models.control_systems import f_dubins, f_dubins_grad
from models.loss_utils import rotation_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(precision=8)


def get_arch(x_dim, u_dim, nonlinearity, system_config):
    if system_config is not None:
        model = FKNet(x_dim + u_dim, 64, x_dim, nonlinearity=nonlinearity,
                out_scale=system_config.fk_out_scale, scale_dims=system_config.fk_scale_dims).to(device)
    else:
        model = FKNet(x_dim + u_dim, 64, x_dim, nonlinearity=nonlinearity).to(device)

    return model


def learn_model(x_dim, u_dim, x_lims, u_lims, f_dynamics, use_grads=False,
                nonlinearity='sine', ori_dims=None, get_loss=False, num_iters=50000,
                system_config=None, save_folder=None):

    batch_size = 4096

    # Nonlinearity
    # nonlinearity = 'sine'
    # nonlinearity = 'relu'

    dims = torch.arange(x_dim).tolist()
    if ori_dims is None:
        pos_dims = dims
    else:
        pos_dims = list(filter(lambda z: z not in ori_dims, dims))

    train_with_jacobian = use_grads
    disturbance = None

    model = get_arch(x_dim, u_dim, nonlinearity, system_config)
    
    mse_loss = nn.MSELoss()

    # opt = optim.Adam(model.parameters(), lr=0.0001)
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

    losses = []

    with trange(0, num_iters, desc="Epochs") as eps:
        for ii in eps:

            xs = torch.FloatTensor(batch_size, x_dim).uniform_(x_lims[0], x_lims[1]).to(device)
            us = torch.FloatTensor(batch_size, u_dim).uniform_(u_lims[0], u_lims[1]).to(device)

            if train_with_jacobian:
                xs.requires_grad_()
                us.requires_grad_()

            xus = torch.cat((xs, us), -1)

            ys_pred = model(xus)
            ys_true = f_dynamics(xs, us)

            if disturbance is not None:
                ys_true[:, :2] += disturbance(xs[:, 0], xs[:, 1])

            opt.zero_grad()


            # if f_dynamics == f_dubins:
            #     loss_pos = mse_loss(ys_pred[:, :2], ys_true[:, :2])
            #     loss_ori = rotation_loss(ys_pred[:, 2], ys_true[:, 2])
            #     loss = loss_pos + loss_ori
            if ori_dims is not None:
                loss_pos = mse_loss(ys_pred[:, pos_dims], ys_true[:, pos_dims])
                loss_ori = rotation_loss(ys_pred[:, ori_dims].reshape(-1), ys_true[:, ori_dims].reshape(-1))
                loss = loss_pos + loss_ori
            else:
                loss = mse_loss(ys_pred, ys_true)

            if train_with_jacobian:
                ys_jac_true = torch.cat((batch_jacobian(ys_true, xs), batch_jacobian(ys_true, us)), -1)
                ys_jac_pred = batch_jacobian(ys_pred, xus)

                loss = loss + (ys_jac_pred - ys_jac_true).pow(2).mean()

            loss.backward()

            opt.step()

            if ii and ii % (num_iters // 100) == 0:
                scheduler.step()

            with torch.no_grad():
                eps.set_postfix(loss=(loss.detach().cpu().item()), lr=opt.param_groups[0]["lr"])
                losses.append(loss.detach().cpu().item())


    with torch.no_grad():

        save_folder = 'results/eval' if save_folder is None else save_folder
        os.makedirs(save_folder, exist_ok=True)

        file_suffix = get_suffix(f_dynamics, nonlinearity, use_grads)

        torch.save(model.state_dict(),
                   os.path.join(save_folder, f'systemid_model_{file_suffix}.pth'))

        model.eval()

        xs = torch.FloatTensor(batch_size, x_dim).uniform_(x_lims[0], x_lims[1]).to(device)
        us = torch.FloatTensor(batch_size, u_dim).uniform_(u_lims[0], u_lims[1]).to(device)
        
        xus = torch.cat((xs, us), -1)

        disturb_field = torch.zeros(batch_size, x_dim).to(device)
        if disturbance is not None:
            disturb_field = torch.cat((disturbance(xs[:, 0], xs[:, 1]), torch.zeros(batch_size, 1).to(device)), -1)

        res = f_dynamics(xs, us) + disturb_field - model(xus)
        
        error_norm = res.norm(2, -1)
        print(
            "Mean:", error_norm.mean().item(),
            "Median:", error_norm.median().item(), 
            "Max:", error_norm.max().item())

    if get_loss:
        return model, losses

    return model


def get_suffix(f_dynamics, nonlinearity, use_grads):
    wjac_sfx = '_wjac' if use_grads else ''

    sfx = f'{f_dynamics.__name__}_{nonlinearity}{wjac_sfx}'
    return sfx


if __name__ == "__main__":
    from models.control_systems import f_cartpole, f_quadrotor
    # fk_model = learn_model(4, 2, x_lims=[-5, 5], u_lims=[-3, 3], f_dynamics=f_cartpole, ori_dims=[2])
    # fk_model = learn_model(4, 2, x_lims=[-5, 5], u_lims=output_scaling, f_dynamics=f_cartpole, ori_dims=[2], use_grads=True)
    # fk_model = learn_model(4, 2, x_lims=[-5, 5], u_lims=[-3, 3], f_dynamics=f_cartpole, ori_dims=None)

    from system_config import SystemConfig
    sys_config = SystemConfig.get_config(system_name='quadrotor')()
    fk_model, losses = learn_model(12, 4, x_lims=[-5, 5], u_lims=[-100, 100], f_dynamics=f_quadrotor, ori_dims=None, use_grads=0, num_iters=100000, get_loss=True, system_config=sys_config)

    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.plot(losses)
    ax.set_title('Losses')
    ax.set_xlabel('Epochs')
    ax.set_yscale('log')

    plt.show()
