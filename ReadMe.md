# CDC2024: Stable Neural Differential Dynamics
<!-- To myself: a clean repo for uploading the code -->
This repository contains the code for our CDC2024 submission: *Jing Cheng, Ruigang Wang, and Ian R. Manchester, Learning Stable and Passive Neural Differential Equations.*

This code has been tested with Python 3.8.10. The figures are generated with MATLAB R2022b.

## Installation

We assume any version later than Python 3.8.10 should work, also please install the packages listed in `requirements.txt`

## Usage
To train the models, run following script to train and save the models: 

`train.py`

The models will be saved in `./results/pendulum2/` dir. 

Then the simulation error is calculated via: 

`simulation_error.py`

The trjectory of physical model will be at `./dataset/p-physics-2.npy`, and all the results from neural dynamics are in `./results/simu_error/`

## Reproduce results
- `loss_trainsize_plot.m` gives the loss v.s. training size figure, data from training with diffrerent training sizes. 
- `mu_nu_tuning.m` gives the loss v.s. $\mu / \nu$ figure, data from training with different parameters $\mu$ and $\nu$ 
- `loss_plot.m` gives the learning loss figure
- `trajectory_plot4.m` gives the simulation trajectory (note: the trajectory figure in the paper was plot with two close initial conditions to better illustrate our strength. Here we just show two batches)
- `error_plot.m` gives the simulation error figure

## Other things to tune/do
Run the follow to test a few training sizes:

`./train_check_trainsize`

## Contact
For any questions or bugs, please raise an issue or contact Ruigang Wang or Jing (Johnny) Cheng via: ruigang.wang, jing.cheng@sydney.edu.au
