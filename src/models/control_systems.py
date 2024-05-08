import torch
import torch.nn.functional as F
import numpy as np

v_max = 1 # [m/s]
min_radius = .5 # [m]


class ControlSystem:
    @staticmethod
    def get_system(system_name: str):
        if system_name == 'dubins': return f_dubins
        elif system_name == 'reedsshepp': return f_reedsshepp
        elif system_name == 'quadrotor': return f_quadrotor
        elif system_name == 'acrobot': return f_acrobot
        elif system_name == 'cartpole': return f_cartpole
        elif system_name == 'single_integrator': return f_singleintegrator
        else: raise NotImplementedError()


def f_singleintegrator(xs, us):
    return us


def f_dubins(xs, us, vs=None):

    if vs is None:
        vs = v_max
    else:
        vs = vs.clamp(0, 1)

    x_dots = torch.stack([
        vs * torch.cos(xs[..., 2]),
        vs * torch.sin(xs[..., 2]),
        vs * us.clamp(-1, 1) / min_radius
    ]).movedim(0, -1).to(xs.device)

    return x_dots


def f_dubins_grad(xs, us, vs=None):

    def df_dx(xs, us, vs):
        batch_size = us.shape[0]
        dfdxs = torch.cat([
            torch.zeros(batch_size, 3, 2).to(xs.device),
            torch.stack((-vs * torch.sin(xs[:, 2]), vs * torch.cos(xs[:, 2]), torch.zeros(batch_size).to(xs.device))).t().unsqueeze(-1)
            ], -1)
        return dfdxs

    def df_du(xs, us, vs):
        batch_size = us.shape[0]
        dfdus = torch.cat([
            torch.zeros(batch_size, 2).to(xs.device),
            (vs / min_radius).unsqueeze(-1)
        ], -1).unsqueeze(-1)
        return dfdus
    
    if vs is None:
        vs = torch.ones_like(us) * v_max

    dfdx = df_dx(xs, us, vs)
    dfdu = df_du(xs, us, vs)
    dfdxu = torch.cat((dfdx, dfdu), -1)
    return dfdxu


def f_reedsshepp(xs, us):
    x_dots = torch.stack([
        us[..., 0] * torch.cos(xs[..., 2]),
        us[..., 0] * torch.sin(xs[..., 2]),
        us[..., 0] * us[..., 1] / min_radius
    ]).movedim(0, -1).to(xs.device)

    return x_dots

def f_quadrotor(xs, controls):

    mass = 1.
    gravity = 9.81

    uu = controls[..., :1]

    # Differential equations
    dpose = xs[..., 6:]

    angles = xs[..., 3:6]
    s_psi, c_psi = torch.sin(angles[..., 0:1]), torch.cos(angles[..., 0:1])
    s_theta, c_theta = torch.sin(angles[..., 1:2]), torch.cos(angles[..., 1:2])
    s_phi, c_phi = torch.sin(angles[..., 2:3]), torch.cos(angles[..., 2:3])

    ddpos_x = (uu / mass) * (s_psi * s_phi + c_psi * s_theta * c_phi)
    ddpos_y = (uu / mass) * (-c_psi * s_phi + s_psi * s_theta * c_phi)
    ddpos_z = (uu / mass) * (c_theta * c_phi) - gravity
    ddori = controls[..., 1:]

    ddpose = torch.cat([ddpos_x, ddpos_y, ddpos_z, ddori], -1)

    dxdt = torch.cat([dpose, ddpose], -1)

    return dxdt

def f_acrobot(xs, controls):

    state = xs
    action = controls

    m1 = 1.  # [m]
    m2 = 1.  # [m]
    l1 = 1.  #: [kg] mass of link 1
    lc1 = 1.  #: [kg] mass of link 2
    lc2 = 0.5  #: [m] position of the center of mass of link 1
    I1 = 0.5  #: [m] position of the center of mass of link 2
    I2 = 1.  #: moments of inertia for both links
    g = 9.8
    theta1,theta2,dtheta1,dtheta2 = state[...,0],state[...,1],state[...,2],state[...,3]
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - np.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2) \
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - np.pi / 2) + phi2
    ddtheta2 = (action[...,0] + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    
    ddtheta1 = -(action[...,1] + d2 * ddtheta2 + phi1) / d1

    return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], -1)


def f_cartpole(xs, us):

    state = xs
    action = us

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    length = 1.0  # actually half the pole's length
    total_mass = (masspole + masscart)
    polemass_length = (masspole * length)
    force_mag = 3.0

    _, x_dot, theta, theta_dot = state[...,0], state[...,1], state[...,2], state[...,3]
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    force = action[...,0] * force_mag
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / \
        (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass

    return torch.stack([x_dot,xacc,theta_dot,thetaacc],-1)
