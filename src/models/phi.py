import torch
import torch.nn as nn
import copy

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    # ## TODO: test the following expression instead
    # return torch.log(1 + torch.exp(2.0 * x)) - x
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))
    # return torch.log(torch.exp(x) + torch.exp(-x)) # numerically unstable

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, n_layers=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param n_layers: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if n_layers < 2:
            print("n_layers must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.n_layers = n_layers
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(n_layers-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.n_layers-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.n_layers):
            x = x + self.h * self.act(self.layers[i](x))

        return x



class Phi(nn.Module):
    def __init__(self, n_layers, m, d, r=10, alph=[1.0] * 6):
        """
            neural network approximating Phi
            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param n_layers:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        self.m    = m
        self.n_layers  = n_layers
        self.d    = d
        self.alph = alph

        r = min(r,d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d, m, n_layers=n_layers)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)



    def forward(self, x):
        """ calculating Phi(s, theta) """

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A
        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)


    def get_grad(self,x):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi)
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d+1
        :return: gradient of Phi
        """

        # assumes specific N.act as the antiderivative of tanh
        N    = self.N
        symA = torch.matmul(self.A.t(), self.A)
        u = [] # hold the u_0,u_1,...,u_M for the forward pass

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.n_layers):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        accGrad = 0.0 # accumulate the gradient as we step backwards through the network
        # compute analytic gradient and fill z
        for i in range(N.n_layers-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.n_layers-1:
                term = self.w.weight.t()
            else:
                term = accGrad # z_{i+1}

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            accGrad = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        tanhopen = torch.tanh(opening)  # act'( K_0 * S + b_0 )
        # z_0 = K_0' diag(...) z_1
        accGrad = torch.mm( N.layers[0].weight.t() , tanhopen.t() * accGrad )
        grad = accGrad + torch.mm(symA, x.t() ) + self.c.weight.t()

        return grad.t()


class SineResNN(nn.Module):
    def __init__(self, d, m, n_layers=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param n_layers: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if n_layers < 2:
            print("n_layers must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.n_layers = n_layers
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(n_layers-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = torch.sin
        self.h = 1.0 / (self.n_layers-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.n_layers):
            x = x + self.h * self.act(self.layers[i](x))

        return x


class SinePhi(nn.Module):
    def __init__(self, n_layers, m, d, r=10):
        """
            neural network approximating Phi
            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param n_layers:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        print(f'Building {type(self)}')

        self.m    = m
        self.n_layers  = n_layers
        self.d    = d
        self.out_d = 1

        r = min(r, d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear(d+1  , self.out_d  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear(m    , self.out_d  , bias=False)

        self.N = SineResNN(d, m, n_layers=n_layers)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)



    def forward(self, x):
        """ calculating Phi(s, theta) """

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A
        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)


    def get_grad(self, x):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi)
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d+1
        :return: gradient of Phi
        """

        # assumes specific N.act as the antiderivative of cos
        N    = self.N
        symA = torch.matmul(self.A.t(), self.A)
        u = [] # hold the u_0,u_1,...,u_M for the forward pass

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.n_layers):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        accGrad = 0.0 # accumulate the gradient as we step backwards through the network
        # compute analytic gradient and fill z
        for i in range(N.n_layers-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.n_layers-1:
                term = self.w.weight.t()
            else:
                term = accGrad # z_{i+1}

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            accGrad = term + N.h * torch.mm( N.layers[i].weight.t() , torch.cos( N.layers[i].forward(u[i-1]) ).t() * term)

        cosopen = torch.cos(opening)  # act'( K_0 * S + b_0 )
        # z_0 = K_0' diag(...) z_1
        accGrad = torch.mm( N.layers[0].weight.t() , cosopen.t() * accGrad )
        grad = accGrad + torch.mm(symA, x.t() ) + self.c.weight.t()

        return grad.t()


if __name__ == "__main__":
    from diff_ops import gradient

    xd = 3
    net = SinePhi(n_layers=2, m=64, d=xd)

    z = torch.randn(128, xd + 1)

    grad_phi = net.get_grad(z)
    z.requires_grad_()
    y = net(z)
    grad_phi_ = gradient(y, z)

    print((grad_phi_ - grad_phi).norm())
