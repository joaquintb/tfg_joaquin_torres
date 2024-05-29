# Author: Joaquín Torres Bravo
# Date: June 2024
# Bachelor Thesis

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm as tqdm
from scipy.integrate import odeint

# Aux function to return RHS of ODE system
def SIR(x, t, N, beta, gamma):
    S, I, R = x
    xdot = [
        -(beta*S*I)/N,
        (beta*S*I)/N - gamma*I,
        gamma*I
    ]
    return xdot

# Defining the network
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce_S = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.fce_I = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.fce_R = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        S_output = self.fce_S(x)
        I_output = self.fce_I(x)
        R_output = self.fce_R(x)
        return S_output, I_output, R_output

if __name__=="__main__": 
    # Defining the problem
    N = 100  # Total population
    # Initial number of infected and recovered individuals
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially
    S0 = N - I0 - R0
    # Initial state of the system
    X0 = [S0, I0, R0]
    # Parameters
    beta, gamma = 1.5, 0.5
    # A grid of time points (in days)
    finalT = 17.0
    t  = np.linspace(0, finalT, 100)
    # Solving the problem numerically (via odeint numerical solver)
    result = odeint(SIR, X0, t, args=(N, beta, gamma))
    S, I, R = result.T

    # Generating observations
    # Generate 10 equispaced time locations in the domain for fair comparison with ABC-SMC
    obs_ind = np.linspace(0, len(t) - 1, 10, dtype=int)
    S_obs = [S[ind] for ind in obs_ind]
    I_obs = [I[ind] for ind in obs_ind]
    R_obs = [R[ind] for ind in obs_ind]
    # Generate Gaussian noise for S_obs, I_obs, and R_obs
    noise_S = 0.5 * np.random.randn(len(S_obs))
    noise_I = 0.5 * np.random.randn(len(I_obs))
    noise_R = 0.5 * np.random.randn(len(R_obs))
    # Add the noise to the observed values
    S_obs_noise = [S_obs[ind] + noise_S[ind] for ind in range(len(S_obs))]
    I_obs_noise = [I_obs[ind] + noise_I[ind] for ind in range(len(I_obs))]
    R_obs_noise = [R_obs[ind] + noise_R[ind] for ind in range(len(R_obs))]
    # Tensor containing time locations for observations
    t_obs = torch.tensor([t[ind] for ind in obs_ind], dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for S
    u_obs_S = torch.tensor(S_obs_noise,  dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for I
    u_obs_I = torch.tensor(I_obs_noise,  dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for R
    u_obs_R = torch.tensor(R_obs_noise,  dtype=torch.float32).view(-1,1)

    # Tensor of times to train PINN through physics loss
    t_physics = torch.linspace(0, finalT, 500).view(-1, 1).requires_grad_(True)

    # Training setup
    # Initialize PINN
    pinn = FCN(1,1,32,3)
    # Sample initial problems parameters from uniform 0 to 10 for comparison to be fair
    beta = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    # Define optimiser and make problem parameters learnable
    optimiser = torch.optim.Adam(list(pinn.parameters())+[beta,gamma],lr=1e-2)
    # Hyperparameter deciding weight given to fitting observational data
    lambda_weight = 10
    # Number of training iterations
    max_its = 30000

    for i in tqdm.tqdm(range(max_its)):
        # Reset gradient to zero
        optimiser.zero_grad()
        # -----------------------
        #       PHYSICS LOSS
        # -----------------------
        uS, uI, uR = pinn(t_physics)
        # Compute the derivatives with respect to time
        duSdt = torch.autograd.grad(uS, t_physics, torch.ones_like(uS), create_graph=True)[0]
        duIdt = torch.autograd.grad(uI, t_physics, torch.ones_like(uI), create_graph=True)[0]
        duRdt = torch.autograd.grad(uR, t_physics, torch.ones_like(uR), create_graph=True)[0]
        # Compute the physics loss for SIR equations
        phy_loss_S = torch.mean((duSdt + (beta * uS * uI) / N) ** 2)
        phy_loss_I = torch.mean((duIdt - (beta * uS * uI) / N + gamma * uI) ** 2)
        phy_loss_R = torch.mean((duRdt - gamma * uI) ** 2)
        total_physics_loss = phy_loss_S + phy_loss_I + phy_loss_R
        # -----------------------
        #       DATA LOSS
        # -----------------------
        # Compute the PINN output
        uS, uI, uR = pinn(t_obs)
        data_loss_S = torch.mean((uS - u_obs_S)**2)
        data_loss_I = torch.mean((uI - u_obs_I)**2)
        data_loss_R = torch.mean((uR - u_obs_R)**2)
        total_data_loss = data_loss_S + data_loss_I + data_loss_R
        # Compute total loss
        loss = total_physics_loss + lambda_weight*total_data_loss
        # Backpropagate joint loss, take optimiser step
        loss.backward()
        optimiser.step()

    print(beta.item())
    print(gamma.item())