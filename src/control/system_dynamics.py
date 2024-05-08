import torch
from warnings import warn
from torch import cos, sin
from math import pi
from .template import ControlledSystemTemplate


class SystemDynamics:
    @staticmethod
    def get_dynamics(system_name: str):
        if system_name == 'dubins': return Dubins
        elif system_name == 'dubins_varyv': return DubinsVaryv
        elif system_name == 'reedsshepp': return ReedsShepp
        elif system_name == 'quadrotor': return Quadrotor
        elif system_name == 'acrobot': return Acrobot
        elif system_name == 'cartpole': return Cartpole
        elif system_name == 'single_integrator': return SingleIntegrator
        elif system_name == 'learned': return LearnedSystem
        else: raise NotImplementedError()


class SingleIntegrator(ControlledSystemTemplate):
    """
    Single integrator model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)
        self.u_vals.append(u)

        # Differential equations
        self.cur_f = u
        return self.cur_f


class Dubins(ControlledSystemTemplate):
    """
    Dubins car
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rad  = 0.5
        self.v_max  = 1.

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)
        self.u_vals.append(u)

        # Differential equations
        dx = self.v_max * torch.cos(x[..., 2:])
        dy = self.v_max * torch.sin(x[..., 2:])
        dtheta = u * self.v_max / self.rad
        
        self.cur_f = torch.cat([dx, dy, dtheta * torch.ones_like(dx)], -1)
        return self.cur_f


class DubinsVaryv(ControlledSystemTemplate):
    """
    Dubins car with varying speed
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rad  = 0.5

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)
        self.u_vals.append(u)

        # Differential equations
        vs = (u[:, 1:] + 1) / 2

        dx = vs * torch.cos(x[..., 2:])
        dy = vs * torch.sin(x[..., 2:])
        dtheta = u[:, :1] * vs / self.rad
        
        self.cur_f = torch.cat([dx, dy, dtheta * torch.ones_like(dx)], -1)
        return self.cur_f


class ReedsShepp(ControlledSystemTemplate):
    """
    ReedsShepp car
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rad  = 0.5

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)
        self.u_vals.append(u)

        # Differential equations
        vs = u[:, 1:]

        dx = vs * torch.cos(x[..., 2:])
        dy = vs * torch.sin(x[..., 2:])
        dtheta = u[:, :1] * vs / self.rad
        
        self.cur_f = torch.cat([dx, dy, dtheta * torch.ones_like(dx)], -1)
        return self.cur_f


class LearnedSystem(ControlledSystemTemplate):
    """
    Dubins car with varying speed
    """
    def __init__(self, model, dubins_car=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dubins_car = dubins_car

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x)
        self.u_vals.append(u)

        # Differential equations
        if self.dubins_car:
            vs = (u[:, 1:] + 1) / 2
            xus = torch.cat([x, u[:, :1], vs], -1)
        else:
            xus = torch.cat([x, u], -1)

        self.cur_f = self.model(xus)
        return self.cur_f


class Quadrotor(ControlledSystemTemplate):
    """
    Quadrotor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gravity  = 9.81
        self.mass = 1.


    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        control = self._evaluate_controller(t, x)
        self.u_vals.append(control)

        uu = control[:, :1]

        # Differential equations
        dpose = x[:, 6:]

        angles = x[:, 3:6]
        s_psi, c_psi = torch.sin(angles[:, 0:1]), torch.cos(angles[:, 0:1])
        s_theta, c_theta = torch.sin(angles[:, 1:2]), torch.cos(angles[:, 1:2])
        s_phi, c_phi = torch.sin(angles[:, 2:3]), torch.cos(angles[:, 2:3])

        ddpos_x = (uu / self.mass) * (s_psi * s_phi + c_psi * s_theta * c_phi)
        ddpos_y = (uu / self.mass) * (-c_psi * s_phi + s_psi * s_theta * c_phi)
        ddpos_z = (uu / self.mass) * (c_theta * c_phi) - self.gravity
        ddori = control[:, 1:]

        ddpose = torch.cat([ddpos_x, ddpos_y, ddpos_z, ddori], -1)

        self.cur_f = torch.cat([dpose, ddpose], -1)
        return self.cur_f


class Acrobot(ControlledSystemTemplate):
    """
    Acrobot
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LINK_LENGTH_1 = 1.  # [m]
        self.LINK_LENGTH_2 = 1.  # [m]
        self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
        self.LINK_MASS_2 = 1.  #: [kg] mass of link 2
        self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
        self.LINK_MOI = 1.  #: moments of inertia for both links

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        control = self._evaluate_controller(t, x)
        self.u_vals.append(control)

        action = control
        state = x

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        theta1,theta2,dtheta1,dtheta2 = state[...,0],state[...,1],state[...,2],state[...,3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - pi / 2) + phi2
        ddtheta2 = (action[...,0] + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        
        ddtheta1 = -(action[...,1] + d2 * ddtheta2 + phi1) / d1

        return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], -1)


class Cartpole(ControlledSystemTemplate):
    """
    Cartpole
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 1.0  # actually half the pole's length
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 3.0

    def dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations
        control = self._evaluate_controller(t, x)
        self.u_vals.append(control)

        action = control
        state = x

        _, x_dot, theta, theta_dot = state[...,0], state[...,1], state[...,2], state[...,3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        force = action[...,0] * self.force_mag
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        return torch.stack([x_dot,xacc,theta_dot,thetaacc],-1)

