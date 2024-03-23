import numpy as np
import os 
import torch
import torch.nn.functional as F
import scipy.io

from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable

from utils import * 
from pendulum import pendulum_gradient, _redim
from LyaProj import stable_dyn
from HNN import Ham_dyn

def torch_redim(x):
    x = _redim(x.cpu().detach().numpy())
    x = torch.tensor(x)
    if torch.cuda.is_available():
        x.cuda()
    x.requires_grad = True
    return x

def main(arg):
    # create dirctories for saving file
    os.makedirs(args.root_dir, exist_ok=True)

    # initialize and load model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    checkpoint = torch.load(arg.model_dir, map_location=device)
    model = args.model(args)
    model.load_state_dict(checkpoint)
    model.eval()

    # load true physic model
    n = args.n
    h = args.timestep
    physics = pendulum_gradient(n)

    simu_path = Path("dataset") /f"p-physics-{n}.npy"

    # Initial condition
    X_init = np.zeros((args.train_size, 2 * n)).astype(np.float32)
    #   Pick values in range [-pi/8, pi/8] radians; 0 radians/sec
    np.random.seed(seed= arg.seed)
    # X_init[:, :n] = (np.random.rand(args.train_size, n).astype(np.float32) - 0.5) * np.pi/2
    
    
    # # For replot figure 5
    X_init = np.zeros((args.train_size, 2*n), dtype=np.float32)
    X_init[:, 1] = np.linspace(0.6, 0.8, args.train_size)
    X_init[:, 0] = np.linspace(-0.5, -0.3, args.train_size)
    

    if not simu_path.exists(): 
        # Generate data from real physical system
        X_phy = np.zeros((args.steps, *X_init.shape), dtype=np.float32)
        X_phy[0,...] = X_init
        for i in range(1, args.steps):
            # logger.info(f"Timestep {i}")
            k1 = h * physics(_redim(X_phy[i-1,...]))
            k2 = h * physics(_redim(X_phy[i-1,...] + k1/2))
            k3 = h * physics(_redim(X_phy[i-1,...] + k2/2))
            k4 = h * physics(_redim(X_phy[i-1,...] + k3))
            X_phy[i,...] = X_phy[i-1,...] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            assert not np.any(np.isnan(X_phy[i,...]))

        np.save(simu_path, X_phy)

    else: 
        X_phy = np.load(simu_path).astype(np.float32)

    print("Physics done")

    # Generate data from model
    X_nn = torch.tensor(X_phy[0,:,:])
    X_nn = Variable(X_nn)
    if torch.cuda.is_available():
        X_nn.cuda()

    X_nn_np = np.zeros((args.steps, *X_init.shape), dtype=np.float32)
    X_nn_np[0,...] = X_init
    mean_error = np.zeros((args.steps, 1))
    max_error = np.zeros((args.steps, 1))
    min_error = np.zeros((args.steps, 1))
    mean_pos_error = np.zeros((args.steps, 1))
    for i in range(1, args.steps):
        X_nn.requires_grad = True
        k1 = h * model(X_nn)
        k1 = k1.detach()
        k2 = h * model(torch_redim(X_nn + k1/2))
        k2 = k2.detach()
        k3 = h * model(torch_redim(X_nn + k2/2))
        k3 = k3.detach()
        k4 = h * model(torch_redim(X_nn + k3))
        k4 = k4.detach()
        X_nn = X_nn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        X_nn = X_nn.detach()

        x_nn_np = X_nn.cpu().detach().numpy() 
        X_nn_np[i,...] = x_nn_np
        error = np.mean(np.linalg.norm((X_nn_np[i,...]-X_phy[i,...]), axis=-1))
        pos_err = np.mean(np.linalg.norm((X_nn_np[i,:,:n]-X_phy[i,:,:n]), axis=-1))
        dX2 = np.linalg.norm((X_nn_np[i,...]-X_phy[i,...]), axis=-1)
        max_err = np.max(dX2)
        min_err = np.min(dX2)
        max_error[i] = max_err
        min_error[i] = min_err
        mean_error[i] = error
        mean_pos_error[i] = pos_err
        if i % 50 == 0:
            print(f'Step: {i:5} | error: {error:.3f}, pos_error: {pos_err:.3f}')
    print(f"Model pridicting done, average mean error: {np.mean(mean_error):.4f}")

    scipy.io.savemat(f'{args.root_dir}/simu_trajectory.mat',
                    {"x_true": X_phy, 
                      "x_model": X_nn_np, 
                      "timestep": h, "steps": 
                      args.steps, 
                      "mean_error":mean_error,
                      "mean_pos_error":mean_pos_error,
                      "max_error": max_error,
                      "min_error": min_error,
                      })
    print("Saved")

if __name__ == "__main__":
    n = 2
    steps = 1000
    train_size = 100
    timestep = 0.01 

    # ICNN 
    args = {
        "root_dir": "./results/simu_error/icnn_dyn_exp",
        "model_dir": "./results/pendulum2/icnn_dyn_exp/model.ckpt", 
        "seed": 42,
        "n": n,
        "model": stable_dyn,
        "proj_fn": "ICNN",
        "nx": 2*n, 
        "nh": 100,
        "nph": 64,
        "alpha": 0.001,
        "eps": 0.01,
        "rehu_factor": 0.005,
        "scale_fx": False ,
        "train_size": train_size,
        "timestep": timestep,
        "steps": steps, 
    }
    args = Dict2Class(args)
    main(args)

    # MLP
    args = {
        "root_dir": "./results/simu_error/stable_dyn_exp",
        "model_dir": "./results/pendulum2/stable_dyn_exp/model.ckpt", 
        "seed": 42,
        "n": n,
        "model": stable_dyn,
        "proj_fn": "NN-REHU",
        "nx": 2*n, 
        "nh": 100,
        "nph": 64,
        "alpha": 0.001,
        "eps": 0.01,
        "rehu_factor": 0.005,
        "scale_fx": False ,
        "train_size": train_size,
        "timestep": timestep,
        "steps": steps, 
    }
    args = Dict2Class(args)
    main(args)

    # # PD-ICNN (Tried, converging to wrong point)
    # args = {
    #     "root_dir": "./results/simu_error/posedef_icnn_dyn_exp",
    #     "model_dir": "./results/pendulum2/posedef_icnn_dyn_exp/model.ckpt", 
    #     "seed": 42,
    #     "n": n,
    #     "model": stable_dyn,
    #     "proj_fn": "PSICNN",
    #     "nx": 2*n, 
    #     "nh": 100,
    #     "nph": 64,
    #     "alpha": 0.001,
    #     "eps": 0.01,
    #     "rehu_factor": 0.005,
    #     "scale_fx": False ,
    #     "train_size": train_size,
    #     "timestep": timestep,
    #     "steps": steps, 
    # }
    # args = Dict2Class(args)
    # main(args)

    # Ham dyn
    args = {
        "root_dir": "./results/simu_error/Ham_dyn_exp",
        "model_dir": "./results/pendulum2/Ham_dyn_exp/model.ckpt", 
        "seed": 42,
        "n": n,
        "model": Ham_dyn,
        "nx": 2*n, 
        "nh": 32,
        "nph": 90,
        "mu": 0.1,
        "nu": 2.,
        "depth": 2,
        "eps": 1e-2,
        "train_size": train_size,
        "timestep": timestep,
        "steps": steps, 
    }
    args = Dict2Class(args)
    main(args)
    print("Done Done")


