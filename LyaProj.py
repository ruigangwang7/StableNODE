import torch
import torch.nn.functional as F
from torch import nn
from ICNN import * 
from utils import * 

class Dynamics(nn.Module):
    def __init__(self, fhat, V, alpha=0.01, scale_fx: bool = False):
        super().__init__()
        self.fhat = fhat
        self.V = V
        self.alpha = alpha
        self.scale_fx = scale_fx

    def forward(self, x):
        fx = self.fhat(x)
        if self.scale_fx:
            fx = fx / fx.norm(p=2, dim=1, keepdim=True).clamp(min=1.0)

        Vx = self.V(x)
        gV = torch.autograd.grad([a for a in Vx], [x], create_graph=True, only_inputs=True)[0]
        rv = fx - gV * (F.relu((gV*fx).sum(dim=1) + self.alpha*Vx[:,0])/(gV**2).sum(dim=1))[:,None]

        return rv
    
def stable_dyn(args):

    nx, nh, nph = args.nx, args.nh, args.nph

    fhat = nn.Sequential(
        nn.Linear(nx, nh), nn.ReLU(),
        nn.Linear(nh, nh), nn.ReLU(),
        nn.Linear(nh, nx)
    )

    if args.proj_fn == "PSICNN":
        V = PosDefICNN([nx, nph, nph, 1], eps=args.eps, negative_slope=0.3)
    elif args.proj_fn == "ICNN":
        V = ICNN([nx, nph, nph, 1])
    elif args.proj_fn == "PSD":
        V = MakePSD(ICNN([nx, nph, nph, 1]), nx, eps=args.eps, d=1.0)
    elif args.proj_fn == "PSD-REHU":
        V = MakePSD(ICNN([nx, nph, nph, 1], activation=ReHU(float(args.rehu_factor))), nx, eps=args.eps, d=1.0)
    elif args.proj_fn == "NN-REHU":
        seq = nn.Sequential(
                nn.Linear(nx, nph,), nn.ReLU(),
                nn.Linear(nph, nph), nn.ReLU(),
                nn.Linear(nph, 1), ReHU(args.rehu_factor)
            )
        V = MakePSD(seq, nx, eps=args.eps, d=1.0)
    elif args.proj_fn == "EndPSICNN":
        V = nn.Sequential(
            nn.Linear(nx, nph, bias=False), nn.LeakyReLU(),
            nn.Linear(nph, nx, bias=False), nn.LeakyReLU(),
            PosDefICNN([nx, nph, nph, 1], eps=args.eps, negative_slope=0.3)
        )
    elif args.proj_fn == "NN":
        V = nn.Sequential(
            nn.Linear(nx, nph,), nn.ReLU(),
            nn.Linear(nph, nph), nn.ReLU(),
            nn.Linear(nph, 1)
        )

    model = Dynamics(fhat, V, alpha=args.alpha, scale_fx=args.scale_fx)

    return model 

if __name__ == "__main__":
    n = 2 
    batch_size = 5
    args = {
        "proj_fn": "ICNN",
        "nx": 2*n, 
        "nh": 64,
        "nph": 64,
        "alpha": 0.01,
        "eps": 0.01,
        "rehu_factor": 0.01,
        "scale_fx": False 
    }
    args = Dict2Class(args)
    model = stable_dyn(args)
    x = torch.rand((batch_size, 2*n), requires_grad=True)
    y = model(x)
    print(y)