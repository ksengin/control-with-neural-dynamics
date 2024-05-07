import torch
from torch import nn

class FKNet(nn.Module):
    """ Fully connected network for FK model
    """
    def __init__(self, input_dim, hidden_size, output_dim, nonlinearity, out_scale: float = 1., scale_dims=None):
        super(FKNet, self).__init__()
        print(f"Building {type(self)}")
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

        self.scale_dims = scale_dims
        self.out_scale = out_scale

        self.u_dim = input_dim - output_dim

        if nonlinearity == 'relu':
            self.act_fn = nn.ReLU()
        elif nonlinearity == 'sine':
            self.act_fn = torch.sin
        elif nonlinearity == 'tanh':
            self.act_fn = nn.Tanh()
        else:
            raise NotImplementedError()

    def forward(self, x: torch.tensor):
        x[..., -self.u_dim:] /= self.out_scale

        h1 = self.act_fn(self.fc1(x))
        h2 = self.act_fn(self.fc2(h1))

        out = self.fc3(h2)

        if self.scale_dims is None:
            out = out * self.out_scale
        else:
            out[..., self.scale_dims] = out[..., self.scale_dims] * self.out_scale

        return out
