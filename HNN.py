import torch
import torch.nn.functional as F
from torch import nn
from PLNet import * 
from utils import * 

class HNN(nn.Module):
    def __init__(self, JR, H, eps = 0.001):
        super().__init__()
        self.H = H
        self.eps = eps
        self.JR = JR

    def forward(self, x):
        batch, nx = x.shape
        Hx = self.H(x)
        gH = torch.autograd.grad([a for a in Hx], [x], create_graph=True, only_inputs=True)[0]
        JR = self.JR(x)
        J, R = JR[:, : nx ** 2], JR[:, nx ** 2 :]
        # J = self.J(x) 
        J = torch.reshape(J, (batch, nx, nx))
        J = J - J.transpose(1,2)
        # R = self.R(x)
        R = torch.reshape(R, (batch, nx, nx))
        R = R @ R.transpose(1,2)
        gH = gH.reshape(batch, 1, nx)
        rv = gH @ (J - R) - self.eps * gH 
        rv = rv.reshape(batch, nx)
        return rv
    
def Ham_dyn(args):
    nx = args.nx 
    nph = args.nph
    JR = nn.Sequential(
        nn.Linear(nx, nph), nn.Tanh(),
        nn.Linear(nph, nph), nn.Tanh(),
        nn.Linear(nph, 2 * (nx ** 2))
    )
    # R = nn.Sequential(
    #     nn.Linear(nx, nph), nn.ReLU(),
    #     nn.Linear(nph, nph), nn.ReLU(),
    #     nn.Linear(nph, nx ** 2)
    # )
    net = BiLipNet(args.nx, [args.nh]*args.depth, mu=args.mu, nu=args.nu)
    H = PLNet(net, use_bias=False)
    model = HNN(JR, H, eps=args.eps)
    return model 

if __name__ == "__main__":
    n = 2 
    batch_size = 5
    args = {
        "nx": 2*n, 
        "nh": 32,
        "nph": 32,
        "mu": 0.25,
        "nu": 4.,
        "depth": 3,
        "eps": 0.01
    }
    args = Dict2Class(args)
    model = Ham_dyn(args)
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(nparams)
    x = torch.rand((batch_size, 2*n), requires_grad=True)
    y = model(x)
    print(y)