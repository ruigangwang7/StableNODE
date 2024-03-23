import os 
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from utils import * 
from pendulum import pendulum_dataset
from LyaProj import stable_dyn
from HNN import Ham_dyn
import scipy.io

def train(args):
    # create log file
    os.makedirs(args.root_dir, exist_ok=True)
    logger = Logger(args.root_dir)

    # make train reproducible
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
     
    # load dataset
    trainData, meta_data = pendulum_dataset(args.n, args.train_size, train=True)
    testData, _ = pendulum_dataset(args.n, args.test_size, train=False)
    train_dataloader = DataLoader(trainData, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(testData, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    logger("data loaded.")

    # create model
    Lr = args.learning_rate
    Epochs = args.epochs 
    steps_per_epoch = len(train_dataloader)
    model = args.model(args) 
    if cuda:
        model = model.cuda()
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    optimizer = optim.Adam(model.parameters(), lr=Lr)
    logger(f"Model size: {1e-3*nparams:.1f}K")

    # lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/10.0, Lr/100.])[0]
    Lr_max, Lr_min = Lr, 0.
    lr_schedule = lambda t: 0.5*(Lr_max+Lr_min) + 0.5*(Lr_max-Lr_min)*np.cos(t/Epochs*np.pi)
    train_loss = np.zeros((Epochs, 1))
    test_loss = np.zeros((Epochs, 1))
    learn_rates = np.zeros((Epochs, 1))
    # run train epochs
    for epoch in range(Epochs):
        tloss = []

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
            optimizer.param_groups[0].update(lr=lr)

            x, y = batch
            if cuda:
                x, y = x.cuda(), y.cuda()
            x.requires_grad = True
            yh = model(x)
            loss = F.mse_loss(yh, y)
            loss.backward()
            optimizer.step()
            tloss.append(loss.cpu().item())

        tloss = sum(tloss) / len(tloss)
        train_loss[epoch] = tloss 
        learn_rates[epoch] = lr 
        vloss = []
        model.eval()
        for batch_dix, batch in enumerate(test_dataloader):
            x, y = batch
            if cuda:
                x, y = x.cuda(), y.cuda()
            x.requires_grad = True
            yh = model(x)
            loss = F.mse_loss(yh, y)
            vloss.append(loss.cpu().item())

        vloss = sum(vloss) / len(vloss)
        test_loss[epoch] = vloss 

        logger(f"Epoch: {epoch+1:4d} | tloss: {tloss:.3f}, vloss: {vloss:.3f}, 100lr: {100*lr:.4f}")

    # save results
    torch.save(model.state_dict(), f"{args.root_dir}/model.ckpt")
    scipy.io.savemat(f'{args.root_dir}/loss.mat', 
                     {"tloss":train_loss, "vloss":test_loss, "lr":learn_rates}) 

if __name__ == "__main__":
    n = 2
    trainsize = 2000
    testsize = 500
    epochs = 1000
    seed = 42

    args = {
        "root_dir": "./results/pendulum2/icnn_dyn_exp",
        "seed": seed,
        "n": n,
        "train_size": trainsize,
        "test_size": testsize,
        "model": stable_dyn,
        "proj_fn": "ICNN",
        "nx": 2*n, 
        "nh": 100,
        "nph": 64,
        "alpha": 0.001,
        "eps": 0.01,
        "rehu_factor": 0.005,
        "scale_fx": False ,
        "train_batch_size": 200,
        "test_batch_size": testsize,
        "learning_rate": 1e-2,
        "epochs": epochs
    }
    args = Dict2Class(args)
    train(args)

    args = {
        "root_dir": "./results/pendulum2/stable_dyn_exp",
        "seed": seed,
        "n": n,
        "train_size": trainsize,
        "test_size": testsize,
        "model": stable_dyn,
        "proj_fn": "NN-REHU",
        "nx": 2*n, 
        "nh": 100,
        "nph": 64,
        "alpha": 0.001,
        "eps": 0.01,
        "rehu_factor": 0.005,
        "scale_fx": False ,
        "train_batch_size": 200,
        "test_batch_size": testsize,
        "learning_rate": 1e-2,
        "epochs": epochs
    }
    args = Dict2Class(args)
    train(args)

    args = {
        "root_dir": "./results/pendulum2/Ham_dyn_exp",
        "seed": seed,
        "n": n,
        "train_size": trainsize,
        "test_size": testsize,
        "model": Ham_dyn,
        "nx": 2*n, 
        "nh": 32,
        "nph": 90,
        "mu": 0.1,
        "nu": 2.0,
        "depth": 2,
        "eps": 1e-2,
        "train_batch_size": 200,
        "test_batch_size": testsize,
        "learning_rate": 1e-2,
        "epochs": epochs
    }
    args = Dict2Class(args)
    train(args)
    
    # args = {
    #     "root_dir": "./results/pendulum2/posedef_icnn_dyn_exp",
    #     "seed": seed,
    #     "n": n,
    #     "train_size": trainsize,
    #     "test_size": testsize,
    #     "model": stable_dyn,
    #     "proj_fn": "PSICNN",
    #     "nx": 2*n, 
    #     "nh": 100,
    #     "nph": 64,
    #     "alpha": 0.001,
    #     "eps": 0.01,
    #     "rehu_factor": 0.005,
    #     "scale_fx": False ,
    #     "train_batch_size": 200,
    #     "test_batch_size": testsize,
    #     "learning_rate": 1e-2,
    #     "epochs": epochs
    # }
    # args = Dict2Class(args)
    # train(args)