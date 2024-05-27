# Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm as tqdm
import time

# Solving the system
def LV(x,a,b,c,d):    
    xdot = np.array([a*x[0] - b*x[0]*x[1], d*x[0]*x[1] - c*x[1]])
    return xdot
def RK4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx,nt))
    x[:,0] = x0
    for k in range(nt-1):
        k1 = dt*f(t[k], x[:,k])
        k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2)
        k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2)
        k4 = dt*f(t[k] + dt, x[:,k] + k3)
        dx=(k1 + 2*k2 + 2*k3 +k4)/6
        x[:,k+1] = x[:,k] + dx;  
    return x, t

# Defining the network
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

# Training function
def train_pinn(pinn, optimizer, t_physics, t_obs, u_obs_x, u_obs_y, target_params, tolerance, lambda1=1e5):
    
    a = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    b = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    c = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    d = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)

    optimizer.add_param_group({'params': [a, b, c, d]})

    target_a, target_b, target_c, target_d = target_params
    start_time = time.time()
    
    for i in tqdm.tqdm(range(20000)):
        optimizer.zero_grad()

        u1, u2 = pinn(t_physics)
        du1dt = torch.autograd.grad(u1, t_physics, torch.ones_like(u1), create_graph=True)[0]
        du2dt = torch.autograd.grad(u2, t_physics, torch.ones_like(u2), create_graph=True)[0]
        phy_loss_x = torch.mean((du1dt - a*u1 + b*u1*u2) ** 2)
        phy_loss_y = torch.mean((du2dt + c*u2 - d*u1*u2) ** 2)
        total_physics_loss = phy_loss_x + phy_loss_y

        u1, u2 = pinn(t_obs)
        data_loss_x = torch.mean((u1 - u_obs_x) ** 2)
        data_loss_y = torch.mean((u2 - u_obs_y) ** 2)
        total_data_loss = data_loss_x + data_loss_y

        loss = total_physics_loss + lambda1 * total_data_loss
        loss.backward()
        optimizer.step()

        if (abs(a.item() - target_a) < tolerance and 
            abs(b.item() - target_b) < tolerance and 
            abs(c.item() - target_c) < tolerance and 
            abs(d.item() - target_d) < tolerance):
            break

    execution_time = time.time() - start_time
    return execution_time, i, (a.item(), b.item(), c.item(), d.item())

if __name__ == "__main__":
    # Defining the problem
    a = 1
    b = 1
    c = 1
    d = 1
    f= lambda t,x : LV(x,a,b,c,d)         # lambda is an anonymous function which can take may inputs but returns one output. Same case is with MATLAB denoted by @.
    x0 = np.array([0.5,1])                # initial condition    
    # Solving the problem
    t0 = 0                                # time unit is second
    tf = 15
    num_points = 1000
    dt = (tf - t0) / (num_points - 1)
    x, t = RK4(f, x0, t0, tf, dt)
    times =[1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4]
    for time in times:
        print(time in t)
    
    exit()
    # Generating obs data
    random_indices = np.random.choice(len(t), size=100, replace=False)
    x_obs = [x[0][ind] for ind in random_indices]
    y_obs = [x[1][ind] for ind in random_indices]
    # Generate Gaussian noise for x_obs and y_obs
    noise_x = 0.03 * np.random.randn(len(x_obs))
    noise_y = 0.03 * np.random.randn(len(y_obs))
    # Add the noise to the observed values
    x_obs_noise = [x_obs[ind] + noise_x[ind] for ind in range(len(x_obs))]
    y_obs_noise = [y_obs[ind] + noise_y[ind] for ind in range(len(y_obs))]

    # Tensors
    t_obs = torch.tensor([t[ind] for ind in random_indices], dtype=torch.float32).view(-1,1)
    u_obs_x = torch.tensor(x_obs_noise,  dtype=torch.float32).view(-1,1) # Noise?
    u_obs_y = torch.tensor(y_obs_noise,  dtype=torch.float32).view(-1,1) 
    t_physics = []
    for i in range(1000):
        if i % 20 == 0:
            t_physics.append(t[i])
    t_physics = torch.tensor(t_physics, dtype=torch.float32).view(-1,1).requires_grad_(True)

    # Initializing 
    pinn = FCN(1,1,32,3)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    target_params = (1.0, 1.0, 1.0, 1.0)
    tolerance = 0.02

    exec_time, end_it, final_params = train_pinn(pinn, optimizer, t_physics, t_obs, u_obs_x, u_obs_y, target_params, tolerance)
    print(f"Execution Time: {exec_time} seconds")
    print(f"Took {end_it} iterations")
    print(f"Final Parameters: {final_params}")