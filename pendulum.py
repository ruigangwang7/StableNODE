import os 
import numpy as np
import torch
from sympy import Dummy, lambdify, symbols
from sympy.physics import mechanics
from typing import Sequence

# Modified from: http://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/
def pendulum_gradient(n, lengths=None, masses=1, friction=0.3):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model
    
    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass) 
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')
    
    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        # Add damping torque:
        forces.append((Ai, -1 * friction * u[i] * A.z))

        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)

    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y, *a, **kw):
        squeeze = False
        if len(y.shape) == 1:
            squeeze = True
            y = np.expand_dims(y, 0)
        rv = np.zeros_like(y)

        for i in range(y.shape[0]):
            # Assume in rad, rad/s:
            #y = np.concatenate([np.broadcast_to(initial_positions, n), np.broadcast_to(initial_velocities, n)])

            vals = np.concatenate((y[i,:], parameter_vals))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            rv[i,:] = np.array(sol).T[0]

        if squeeze:
            return rv[0,...]
        return rv

    # ODE integration
    return gradient

def _redim(inp):
    vec = np.array(inp)
    # Wrap all dimensions:
    n = vec.shape[1] // 2
    assert vec.shape[1] == n*2

    # Get angular positions:
    pos = vec[:,:n]

    if np.any(pos < -np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos < -np.pi] + np.pi) / (2*np.pi))
        # Scale it back
        pos[pos < -np.pi] = (adj * 2*np.pi) + np.pi
        assert not np.any(pos < -np.pi)

    if np.any(pos >= np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos >= np.pi] - np.pi) / (2*np.pi))
        # Scale it back
        pos[pos >= np.pi] = (adj * 2*np.pi) - np.pi
        assert not np.any(pos >= np.pi)

    vec[:,:n] = pos
    return vec

def pendulum_dataset(
    n: int, 
    data_size: int, 
    train: bool = True,
    lower_energy: bool = False,
    seeds: Sequence[int] = [42, 43]
):
    if train: 
        seed = seeds[0]
        train_str = "train"
    else:
        seed = seeds[1]
        train_str = "test"
    np.random.seed(seed)
    pen_gen = pendulum_gradient(n)
    le_str = "-lowenergy" if lower_energy else ""
    file = f"./dataset/pendulum-{n}{le_str}-{train_str}-size{data_size}-seed{seed}.npz"
    os.makedirs("./dataset", exist_ok=True)
    if os.path.isfile(file):
        data = np.load(file)
        x, y = data["x"], data["y"]
    else:
        if lower_energy:
            x = np.zeros((data_size, 2*n))
            x[:, :n] = (np.random.rand(data_size, n).astype(np.float32) - 0.5) * np.pi / 2
        else:
            x = (np.random.rand(data_size, 2*n).astype(np.float32) - 0.5) * 2 * np.pi
        y = pen_gen(x)
        np.savez(file, x=x, y=y)
    
    rv = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    meta_data ={
        "num_link": n,
        "redim": _redim,
        "func": pen_gen 
    }

    return rv, meta_data

if __name__ == "__main__":
    train_size = 2000
    test_size = 1000
    n = 2 
    seeds = [1, 2]
    rv, meta_data = pendulum_dataset(n, train_size, train=True, seeds=seeds)
    rv, _ = pendulum_dataset(n, test_size, train=False, seeds=seeds)
    rv, _ = pendulum_dataset(n, train_size, train=True, seeds=seeds)
    rv, _ = pendulum_dataset(n, test_size, train=False, seeds=seeds)
