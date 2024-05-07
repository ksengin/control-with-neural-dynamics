import os
import json
import configargparse
import numpy as np
import time
from tqdm import trange
from math import pi
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from control.system_dynamics import SystemDynamics
from control.cost import IntegralCost
from control.controllers import *

from models.phi import Phi, SinePhi
from models.diff_ops import gradient, jacobian
from models.control_systems import ControlSystem
from system_config import SystemConfig
from system_id import learn_model, get_arch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_results(args, robot, init_dist, model_dir, suffix_str):
    def plot_robot_trajs(t_span, traj):
        plt.cla()
        x_dim = min(traj.shape[-1], 5)
        fig, axs = plt.subplots(1, x_dim, figsize=(15,4))
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
        for i in range(len(x0)):
            for j in range(x_dim):
                axs[j].plot(t_span.cpu(), traj[:,i,j].detach().cpu(), colors[j % len(colors)], alpha=.3)
        for i in range(x_dim):
            axs[i].set_xlabel(r'Time [s]')
            axs[i].set_ylabel(r'$x$' + f'{i}')
    t0 = 0.
    tf = args.t_final

    dt = args.dt # step size
    steps = int((tf - t0)/dt) + 1
    t_span = torch.linspace(t0, tf, steps).to(device)

    robot.solver = 'dopri5'

    # Forward propagate some trajectories 
    x0 = init_dist.sample((100,)).to(device)#*0.8

    traj, acts = robot(x0, t_span)
    
    plot_robot_trajs(t_span, traj)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'result_{suffix_str}.pdf'))
    plt.close()


def main(args):

    n_iters = args.n_iters
    log_freq = 10

    t0 = 0.
    tf = args.t_final

    dt = args.dt # step size
    steps = int((tf - t0)/dt) + 1
    t_span = torch.linspace(t0, tf, steps).to(device)

    ####
    sys_config = SystemConfig.get_config(system_name=args.system)()
    x_dim = sys_config.x_dim
    u_dim = sys_config.u_dim

    x_star = sys_config.x_star
    u_star = sys_config.u_star

    w_P = sys_config.w_P if args.w_P is None else args.w_P
    w_Q = sys_config.w_Q if args.w_Q is None else args.w_Q
    w_R = sys_config.w_R if args.w_R is None else args.w_R
    cost = IntegralCost(x_star=x_star, u_star=u_star, P=w_P, Q=w_Q, R=w_R)

    init_dist = sys_config.init_dist
    ####

    # Controller
    output_scaling = sys_config.output_scaling * args.scale_action # controller limits
    controller = BoxConstrainedController(x_dim, u_dim, constrained=True, output_scaling=output_scaling).to(device)

    model_dir = f'logs/{args.sess}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'session_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log_name = os.path.join(model_dir, 'loss_log.txt')

    # Value network
    if args.value_net == 'phi':
        v_net = Phi(n_layers=args.n_resnet_layers, m=64, d=x_dim)
    else:
        v_net = SinePhi(n_layers=args.n_resnet_layers, m=64, d=x_dim)
    v_net.to(device)


    # Control system and dynamics
    # Will need to use the dynamics eqns if system is not learned or if fkmodel needs to be trained
    if args.archive_fkmodel is None:
        f_system = ControlSystem.get_system(args.system)

    system_name = args.system if not args.learned_fk else 'learned'
    system_dynamics = SystemDynamics.get_dynamics(system_name)

    # Kinematics function network
    if args.learned_fk:
        if args.archive_fkmodel is None:
            fk_model = learn_model(x_dim, u_dim, x_lims=[-5, 5], u_lims=output_scaling, f_dynamics=f_system,
                                use_grads=args.fk_wjac, nonlinearity=args.fk_act_fn, ori_dims=None,
                                num_iters=100000, system_config=sys_config, save_folder=model_dir)
        else:
            fk_model = get_arch(x_dim, u_dim, args.fk_act_fn, sys_config)
            fk_model.load_state_dict(torch.load(args.archive_fkmodel, map_location=device))

        fk_model.to(device)
        fk_model.eval()

    is_dubins = 'dubins' in args.system

    if not args.learned_fk:
        robot = system_dynamics(controller, solver='euler')
    else:
        robot = system_dynamics(fk_model, dubins_car=is_dubins, u=controller, solver='euler')


    # Hyperparameters
    lr = args.start_lr
    epochs = n_iters
    bs = args.batch_size

    opt = torch.optim.Adam(list(v_net.parameters()) + list(controller.parameters()), lr=lr)
    gamma_ = 0.99
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma_)
    sch_rate = int(max(epochs // (np.log(0.001) / np.log(gamma_)), 10))

    if args.archive is not None:
        models = torch.load(args.archive, map_location=device)
        start_iter = models['epoch'] + 1

        v_net.load_state_dict(models['vnet'])
        controller.load_state_dict(models['controller'])
        opt.load_state_dict(models['opt'])
        if args.learned_fk:
            fk_model.load_state_dict(models['fknet'])
    else:
        start_iter = 0

    # Training loop
    losses=[]
    print('Start training')
    with trange(start_iter, epochs, desc="Epochs") as eps:
        for epoch in eps:    
            x0 = init_dist.sample((bs,)).to(device)
            trajectory, actions = robot(x0, t_span)
            
            terminal_cost = torch.norm(w_P * (trajectory[-1] - x_star), p=2, dim=-1)
            running_cost = torch.norm(w_R * (actions - u_star), p=2, dim=-1)
            
            t_span_ = t_span[:, None, None].repeat(1, bs, 1)
            xt = torch.cat((trajectory, t_span_), -1)
            dVdx = v_net.get_grad(xt.view(-1, x_dim + 1)).reshape(t_span.shape[0], bs, -1)
            
            if not args.learned_fk:
                dxdt = f_system(trajectory[:-1], actions)
            else:
                dxdt = fk_model(torch.cat((trajectory[:-1], actions), -1))

            H = running_cost + (dVdx[:-1, :, :x_dim] * dxdt).sum(-1)
            loss_hamiltonian = gradient(H, actions).pow(2).mean()
            
            dVdt = dVdx[:-1, :, -1]
            loss_hjb = (dVdt - H).abs().mean()
            
            loss_end = (v_net(F.pad(trajectory[-1], [0, 1], value=tf)).squeeze() - terminal_cost).abs().mean()
            
            loss_cost = cost(trajectory, actions)
            # loss = loss_cost * 10 + loss_hjb + loss_hamiltonian * 0.01 + loss_end * 0.01
            loss = loss_cost * 10 + loss_hjb + loss_end * 0.01


            if args.use_obstacle:
                loss += sys_config.obstacle_fn(trajectory[..., :sys_config.obstacle_centers.shape[-1]], sys_config.obstacle_centers).mean()
            
            losses.append(loss.detach().cpu().item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            eps.set_postfix(loss=(loss.detach().cpu().item()), cost=loss_cost.detach().cpu().item(),
                            lr=opt.param_groups[0]["lr"])

            if epoch and epoch % sch_rate == 0:
                scheduler.step()

            if epoch and epoch % log_freq == 0:
                with torch.no_grad():
                    message = f'{epoch} - Loss: {loss.cpu().item()}, Cost: {loss_cost.detach().cpu().item()}, LR: {opt.param_groups[0]["lr"]}'
                    
                    if epoch % (args.save_freq * log_freq) == 0:

                        save_dict = {
                            'epoch': epoch,
                            'vnet': v_net.state_dict(),
                            'controller': controller.state_dict(),
                            'opt': opt.state_dict(),
                            'fknet': fk_model.state_dict() if args.learned_fk else None
                        }

                        torch.save(save_dict, os.path.join(model_dir, f'model_{epoch:06d}.pth'))

                    with open(log_name, "a") as log_file:
                        log_file.write('%s\n' % message)

    with torch.no_grad():
        save_dict = {
            'epoch': epoch,
            'vnet': v_net.state_dict(),
            'controller': controller.state_dict(),
            'opt': opt.state_dict(),
            'fknet': fk_model.state_dict() if args.learned_fk else None
        }
        torch.save(save_dict, os.path.join(model_dir, f'model_final.pth'))

        plot_results(args, robot, init_dist, model_dir, f'eval_surrogatefk')

        if 'nimble' not in args.system:
            robot_test = SystemDynamics.get_dynamics(args.system)(None, solver='euler')
            robot_test.u = robot.u

            plot_results(args, robot_test, init_dist, model_dir, f'eval_truefk')

    print('done!')


def get_args():

    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    parser.add_argument('--system', type=str, 
        choices=['single_integrator', 'dubins', 'dubins_varyv', 'reedsshepp', 'quadrotor', 'acrobot', 'cartpole', 'nimblearm', 'nimblearmjoint', 'nimblehusky'])
    parser.add_argument('--fk_act_fn', type=str, default='sine', choices=['relu', 'sine'])
    parser.add_argument('--fk_wjac', action='store_true', help='train fknet with jacobians if true')
    parser.add_argument('--value_net', type=str, default='phi', choices=['phi', 'sinephi'])
    parser.add_argument('--archive', type=str)
    parser.add_argument('--archive_fkmodel', type=str)

    parser.add_argument('--learned_fk', action='store_true', help='use learned dynamics if true')
    parser.add_argument('--init_sampling', type=str, default='fixed', choices=['fixed', 'rand', 'single'])
    parser.add_argument('--sess', type=str, default='dubins')
    parser.add_argument('--scale_action', type=float, default=1., help='scale the action limits, optionally')
    parser.add_argument('--ws_len', type=float, default=5)
    parser.add_argument('--use_obstacle', action='store_true')
    parser.add_argument('--obs_penalty', type=float, default=100.)
    parser.add_argument('--obs_clip', action='store_true', help='clip dx/dt when obstacle penetration if true')
    parser.add_argument('--disturbance', type=str, choices=['turbulence', 'const', 'spiral', 'sinks', 'vortex'])
    parser.add_argument('--k_ext_force', type=float, default=1.)

    parser.add_argument('--x_dim', type=int, default=4)
    parser.add_argument('--u_dim', type=int, default=2)
    parser.add_argument('--t_final', type=float, default=4)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--n_resnet_layers', type=int, default=2)

    # Cost
    parser.add_argument('--w_P', type=float)
    parser.add_argument('--w_Q', type=float)
    parser.add_argument('--w_R', type=float)

    # Hyperparameters
    parser.add_argument('--n_iters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--start_lr', type=float, default=0.01)

    # Logging
    parser.add_argument('--save_freq', type=int, default=100)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(get_args())
