# Author: Joaquín Torres Bravo
# Date: June 2024
# Bachelor Thesis

# Libraries
import numpy as np
import torch
import torch.nn as nn
import tqdm as tqdm

# RK4 to numerically solve the LV system (Vanessa)
def RK4(f, x0, t0, tf, num_points):
    t = np.linspace(t0,tf, num_points)
    dt = (tf - t0) / (num_points - 1)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx,nt))
    x[:,0] = x0 # Initial condition
    for k in range(nt-1):
        k1 = dt*f(t[k], x[:,k])
        k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2)
        k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2)
        k4 = dt*f(t[k] + dt, x[:,k] + k3)
        dx=(k1 + 2*k2 + 2*k3 +k4)/6
        x[:,k+1] = x[:,k] + dx
    return x, t

# PINN model definition
class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch with sinusoidal activation"
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
        self.fce_predator = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.fce_prey = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        prey_output = self.fce_prey(x)
        predator_output = self.fce_predator(x)
        return prey_output, predator_output

if __name__=="__main__": 
    # Setup
    # Defining the problem
    a,b,c,d = 1,1,1,1
    f=lambda t, x: np.array([a*x[0] - b*x[0]*x[1], d*x[0]*x[1] - c*x[1]])
    x0 = np.array([0.5,1]) # IC
    # Solving the problem
    t0, tf = 0, 15
    num_points = 1000
    x, t = RK4(f, x0, t0, tf, num_points)

    # Generating observations
    # Generate 8 equispaced time locations in the domain for fair comparison with ABC-SMC
    obs_ind = np.linspace(0, len(t) - 1, 8, dtype=int)
    # Get solution at those time locations
    x_obs = [x[0][ind] for ind in obs_ind]
    y_obs = [x[1][ind] for ind in obs_ind]
    # Generate Gaussian noise to add to the observations
    noise_x = 0.03 * np.random.randn(len(x_obs))
    noise_y = 0.03 * np.random.randn(len(y_obs))
    # Add the noise to the observed values
    x_obs_noise = [x_obs[ind] + noise_x[ind] for ind in range(len(x_obs))]
    y_obs_noise = [y_obs[ind] + noise_y[ind] for ind in range(len(y_obs))]
    # Tensor containing time locations for observations
    t_obs = torch.tensor([t[ind] for ind in obs_ind], dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for prey
    u_obs_x = torch.tensor(x_obs_noise,  dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for predator
    u_obs_y = torch.tensor(y_obs_noise,  dtype=torch.float32).view(-1,1)

    # Physic loss training points
    # Tensor of times to train PINN through physics loss
    t_physics = torch.linspace(t0, tf, 500).view(-1, 1).requires_grad_(True)

    # Training setup
    # Initialize PINN
    pinn = FCN(1,1,32,3) 
    # Sample initial problems parameters from uniform 0 to 10 for comparison to be fair
    a = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    b = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    c = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    d = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    # Define optimiser and make problem parameters learnable
    optimiser = torch.optim.Adam(list(pinn.parameters())+[a,b,c,d],lr=1e-3)
    # Hyperparameter deciding weight given to fitting observational data
    lambda_weight = 100
    # Number of training iterations
    max_it = 30000

    # Training loop
    for i in tqdm.tqdm(range(max_it)):
        # Reset gradient to zero
        optimiser.zero_grad()
        # -----------------------
        #       PHYSICS LOSS
        # -----------------------
        u1,u2 = pinn(t_physics)
        # Compute the derivatives with respect to time
        du1dt = torch.autograd.grad(u1, t_physics, torch.ones_like(u1), create_graph=True)[0]
        du2dt = torch.autograd.grad(u2, t_physics, torch.ones_like(u2), create_graph=True)[0]
        # Compute the physics loss for Lotka-Volterra equations
        phy_loss_x = torch.mean((du1dt - a*u1 + b*u1*u2) ** 2)
        phy_loss_y = torch.mean((du2dt + c*u2 - d*u1*u2) ** 2)
        # Compute total physics loss
        total_physics_loss = phy_loss_x + phy_loss_y
        # -----------------------
        #       DATA LOSS
        # -----------------------
        # Compute the PINN output
        u1, u2 = pinn(t_obs)
        # Compute the data loss for the first equation (x)
        data_loss_x = torch.mean((u1 - u_obs_x)**2)
        # Compute the data loss for the second equation (y)
        data_loss_y = torch.mean((u2 - u_obs_y)**2)
        # Compute total data loss
        total_data_loss = data_loss_x + data_loss_y
        # Compute total loss
        loss = total_physics_loss + lambda_weight*total_data_loss
        # Backpropagate joint loss, take optimiser step
        loss.backward()
        optimiser.step()
    
    print(a.item())
    print(b.item())
    print(c.item())
    print(d.item())