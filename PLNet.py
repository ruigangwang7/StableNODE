import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np 
from ICNN import ReHU

def cayley(W: torch.Tensor) -> torch.Tensor:
    cout, cin = W.shape
    if cin > cout:
        return cayley(W.T).T
    U, V = W[:cin, :], W[cin:, :]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)
    A = U - U.T + V.T @ V
    iIpA = torch.inverse(I + A)

    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=0)

class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.empty(1).fill_(
            self.weight.norm().item()), requires_grad=True)

        self.Q_cached = None

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        self.Q_cached = None

    def forward(self, X):
        if self.training:
            self.Q_cached = None
            Q = cayley(self.alpha * self.weight / self.weight.norm())
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley(
                        self.alpha * self.weight / self.weight.norm())
            Q = self.Q_cached

        return F.linear(X, Q, self.bias)

class MonLipLayer(nn.Module):
    def __init__(self, 
                 features: int, 
                 unit_features: Sequence[int],
                 mu: float = 0.1,
                 nu: float = 10.):
        super().__init__()
        self.mu = mu
        self.nu = nu  
        self.units = unit_features
        self.Fq = nn.Parameter(torch.empty(sum(self.units), features))
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.empty((1,)))
        nn.init.constant_(self.fq, self.Fq.norm())
        self.by = nn.Parameter(torch.zeros(features))
        Fr, fr, b = [], [], []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz+nz_1)))
            nn.init.xavier_normal_(R)
            r = nn.Parameter(torch.empty((1,)))
            nn.init.constant_(r, R.norm())
            Fr.append(R)
            fr.append(r)
            b.append(nn.Parameter(torch.zeros(nz)))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.b = nn.ParameterList(b)
        # cached weights
        self.Q = None 
        self.R = None 
        self.act = ReHU(1.)

    def forward(self, x):
        sqrt_gam = math.sqrt(self.nu - self.mu)
        sqrt_2 = math.sqrt(2.)
        if self.training:
            self.Q, self.R = None, None 
            Q = cayley(self.fq * self.Fq / self.Fq.norm())
            R = [cayley(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
        else:
            if self.Q is None:
                with torch.no_grad():
                    self.Q = cayley(self.fq * self.Fq / self.Fq.norm())
                    self.R = [cayley(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
            Q, R = self.Q, self.R 

        xh = sqrt_gam * x @ Q.T
        yh = []
        hk_1 = xh[..., :0]
        idx = 0 
        for k, nz in enumerate(self.units):
            xk = xh[..., idx:idx+nz]
            gh = sqrt_2 * self.act (sqrt_2 * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.b[k]) @ R[k]
            # gh = sqrt_2 * F.relu (sqrt_2 * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.b[k]) @ R[k]
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz 
            hk_1 = hk 
        yh.append(hk_1)

        yh = torch.cat(yh, dim=-1)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh @ Q) + self.by 

        return y
    
class BiLipNet(nn.Module):
    def __init__(self,
                 features: int, 
                 unit_features: Sequence[int],
                 mu: float = 0.1,
                 nu: float = 10.,
                 nlayer: int = 1):
        super().__init__()
        self.nlayer = nlayer
        mu = mu ** (1./nlayer)
        nu = nu ** (1./nlayer)
        olayer = [CayleyLinear(features, features) for _ in range(nlayer+1)]
        self.orth_layers = nn.Sequential(*olayer)
        mlayer = [MonLipLayer(features, unit_features, mu, nu) for _ in range(nlayer)]
        self.mon_layers = nn.Sequential(*mlayer)

    def forward(self, x):
        for k in range(self.nlayer):
            x = self.orth_layers[k](x)
            x = self.mon_layers[k](x)
        x = self.orth_layers[self.nlayer](x)
        return x 

class PLNet(nn.Module):
    def __init__(self, bln, use_bias: bool = False):
        super().__init__()
        self.bln = bln 
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((1,))) 

    def forward(self, x):
        x = self.bln(x)
        x0 = self.bln(0.*x)
        y = 0.5 * torch.norm(x-x0, dim=-1) ** 2 
        if self.use_bias:
            y += self.bias
        return y     
    
if __name__ == "__main__":
    batch_size=5
    features = 32
    units = [32, 64, 128]
    mu=0.5
    nu=2.0
    bln = BiLipNet(features, units, mu, nu)
    nparams = np.sum([p.numel() for p in bln.parameters() if p.requires_grad])
    print(nparams)
    model = PLNet(bln)
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(nparams)
    x=torch.randn((batch_size, features))
    y=model(x)