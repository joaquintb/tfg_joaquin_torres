# Author: Joaquín Torres Bravo
# Date: June 2024
# Bachelor Thesis

# Libraries
import numpy as np
import torch
import torch.nn as nn
import tqdm as tqdm
import matplotlib.pyplot as plt
import time as time
import seaborn as sns

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
    
def get_obs_data(t,x,y):
    # Generate 8 equispaced time locations in the domain for fair comparison with ABC-SMC
    obs_ind = np.linspace(0, len(t) - 1, 8, dtype=int)
    # Get solution at those time locations
    x_obs = np.array([x[ind] for ind in obs_ind])
    y_obs = np.array([y[ind] for ind in obs_ind])
    # Generate Gaussian noise to add to the observations
    noise_x = 0.01 * x_obs * np.random.randn(len(x_obs))
    noise_y = 0.01 * y_obs * np.random.randn(len(y_obs))
    # Add the noise to the observed values
    x_obs_noise = [x_obs[ind] + noise_x[ind] for ind in range(len(x_obs))]
    y_obs_noise = [y_obs[ind] + noise_y[ind] for ind in range(len(y_obs))]
    # Tensor containing time locations for observations
    t_obs = torch.tensor([t[ind] for ind in obs_ind], dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for prey
    u_obs_x = torch.tensor(x_obs_noise,  dtype=torch.float32).view(-1,1)
    # Tensor with noisy observations for predator
    u_obs_y = torch.tensor(y_obs_noise,  dtype=torch.float32).view(-1,1)

    return t_obs, u_obs_x, u_obs_y

def simulate(t_obs, u_obs_x, u_obs_y, t_physics, target_params, tol, max_its, lambda_weight):
    # Initialize PINN
    pinn = FCN(1,1,32,3) 
    # Sample initial problems parameters from uniform 0 to 5 for comparison to be fair
    a = torch.nn.Parameter(torch.empty(1).uniform_(0, 5), requires_grad=True)
    b = torch.nn.Parameter(torch.empty(1).uniform_(0, 5), requires_grad=True)
    c = torch.nn.Parameter(torch.empty(1).uniform_(0, 5), requires_grad=True)
    d = torch.nn.Parameter(torch.empty(1).uniform_(0, 5), requires_grad=True)
    # Define optimiser and make problem parameters learnable
    optimiser = torch.optim.Adam(list(pinn.parameters())+[a,b,c,d],lr=1e-2)
    a_t, b_t, c_t, d_t = target_params[0], target_params[1], target_params[2], target_params[3]
    a_s, b_s, c_s, d_s = [], [], [], [] # To keep track of progress
    a_times, b_times, c_times, d_times = [], [], [], [] # To keep track of execution time at each iteration (for plotting purposes)
    a_done, b_done, c_done, d_done = False, False, False, False # Keep track of state of each parameter individually

    # Start timing
    start_time = time.time()

    # Training loop
    for it in range(max_its):
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
        # Record values, stop recording invidually if one of them is done
        if not a_done:
            curr_time = time.time()
            a_times.append(curr_time-start_time)
            a_s.append(a.item())
            a_done = True if abs(a.item() - a_t) < tol else False
        if not b_done:
            curr_time = time.time()
            b_times.append(curr_time-start_time)
            b_s.append(b.item())
            b_done = True if abs(b.item() - b_t) < tol else False
        if not c_done:
            curr_time = time.time()
            c_times.append(curr_time - start_time)
            c_s.append(c.item())
            c_done = True if abs(c.item() - c_t) < tol else False

        if not d_done:
            curr_time = time.time()
            d_times.append(curr_time - start_time)
            d_s.append(d.item())
            d_done = True if abs(d.item() - d_t) < tol else False
        
        if a_done and b_done and c_done and d_done:
            end_time = time.time()
            final_time = end_time - start_time
            return a_s, a_times, b_s, b_times, c_s, c_times, d_s, d_times, final_time, True

    end_time = time.time()
    final_time = end_time - start_time
    return a_s, a_times, b_s, b_times, c_s, c_times, d_s, d_times, final_time, False

def stat_report(execution_times):
    mean = np.mean(execution_times)
    sd = np.std(execution_times)
    median = np.median(execution_times)
    iqr = np.percentile(execution_times, 75) - np.percentile(execution_times, 25)

    print('Printing statistical report of execution time: ')
    print("Mean:", mean)
    print("Standard Deviation:", sd)
    print("Median:", median)
    print("Interquartile Range:", iqr)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(execution_times, color='blue')
    plt.title("Histogram of Execution Times")
    plt.xlabel("Execution Time (s)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.boxplot(execution_times, vert=False)
    plt.title("Box Plot of Execution Times")
    plt.xlabel("Execution Time (s)")

    plt.savefig('./sim_results/execution_time_pinn.png')

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
    # Get observational data from the solution
    t_obs, u_obs_x, u_obs_y = get_obs_data(t, x[0], x[1])
    # Tensor of times to train PINN through physics loss
    num_phys_locs = 700
    t_physics = torch.linspace(t0, tf, num_phys_locs).view(-1, 1).requires_grad_(True)
    # Hyperparameter deciding weight given to fitting observational data
    lambda_weight = 10

    # Simulation setup
    num_sim = 100 # Number of simulations to run
    tol = 0.1 # Tolerance to target needed to stop training
    max_its = 20000
    # Target parameters
    target_params = [a,b,c,d]
    #                                    SIMULATION LOOP
    # --------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))
    ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    fig.suptitle("Parameter trajectories for all simulations")
    total_exec_times = [] # Keep track of execution time of each simulation
    last_a_times, last_b_times, last_c_times, last_d_times = [], [], [], []

    for sim_id in tqdm.tqdm(range(1, num_sim+1)):
        a_s, a_times, b_s, b_times, c_s, c_times, d_s, d_times, final_time, stopped = simulate(t_obs, u_obs_x, u_obs_y, t_physics, target_params, tol, max_its, lambda_weight)
        # print(f'SIM {sim_id}: a={a_s[-1]}, b={b_s[-1]}, c={c_s[-1]}, d={d_s[-1]}, t = {final_time} s')
        total_exec_times.append(final_time)
        last_a_times.append(a_times[-1])
        last_b_times.append(b_times[-1])
        last_c_times.append(c_times[-1])
        last_d_times.append(d_times[-1])
        # Plot trajectories in execution time of each parameter for current simulation
        ax1.plot(a_times, a_s)
        ax2.plot(b_times, b_s)
        ax3.plot(c_times, c_s)
        ax4.plot(d_times, d_s)
    # --------------------------------------------------------------------------------------------------

    #                                    PLOTTING PARAM TRAJ
    # --------------------------------------------------------------------------------------------------
    ax1.set_title("a values")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("a value")
    ax1.set_xlim(0, max(last_a_times))
    ax1.set_ylim(-1,6)
    # Adding horizontal lines for the tolerance range around 'a'
    ax1.axhline(y=a + tol, color='tab:blue', linestyle='dashed', linewidth=1, label=f'a + tol')
    ax1.axhline(y=a - tol, color='tab:red', linestyle='dashed', linewidth=1, label=f'a - tol')
    # Shading the area between 'a + tol' and 'a - tol'
    ax1.fill_between([0,max(last_a_times)], a - tol, a + tol, color='green', alpha=0.3, label='Convergence Region')
    ax1.legend(loc="best")

    ax2.set_title("b values")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("b value")
    ax2.set_xlim(0, max(last_b_times))
    ax2.set_ylim(-1, 6)
    ax2.axhline(y=b + tol, color='tab:blue', linestyle='dashed', linewidth=1, label=f'b + tol')
    ax2.axhline(y=b - tol, color='tab:red', linestyle='dashed', linewidth=1, label=f'b - tol')
    ax2.fill_between([0, max(last_b_times)], b - tol, b + tol, color='green', alpha=0.3, label='Convergence Region')
    ax2.legend(loc="best")

    ax3.set_title("c values")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("c value")
    ax3.set_xlim(0, max(last_c_times))
    ax3.set_ylim(-1, 6)
    ax3.axhline(y=c + tol, color='tab:blue', linestyle='dashed', linewidth=1, label=f'c + tol')
    ax3.axhline(y=c - tol, color='tab:red', linestyle='dashed', linewidth=1, label=f'c - tol')
    ax3.fill_between([0, max(last_c_times)], c - tol, c + tol, color='green', alpha=0.3, label='Convergence Region')
    ax3.legend(loc="best")

    ax4.set_title("d values")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("d value")
    ax4.set_xlim(0, max(last_d_times))
    ax4.set_ylim(-1, 6)
    ax4.axhline(y=d + tol, color='tab:blue', linestyle='dashed', linewidth=1, label=f'd + tol')
    ax4.axhline(y=d - tol, color='tab:red', linestyle='dashed', linewidth=1, label=f'd - tol')
    ax4.fill_between([0, max(last_d_times)], d - tol, d + tol, color='green', alpha=0.3, label='Convergence Region')
    ax4.legend(loc="best")

    plt.tight_layout()
    # plt.show()
    plt.savefig('./sim_results/param_trajec_pinn.png')
    # --------------------------------------------------------------------------------------------------

    stat_report(total_exec_times)