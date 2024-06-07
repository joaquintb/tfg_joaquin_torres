# Author: Joaquín Torres Bravo
# Date: June 2024
# Bachelor Thesis
# Universidad Carlos III de Madrid

# ------------------------------------------------------------------------------------------------------
# Portions of this code are adapted from the work by Ben Moseley, 2022
# Original source is licensed under the MIT License:
# https://github.com/benmoseley/harmonic-oscillator-pinn-workshop/blob/main/PINN_intro_workshop.ipynb
# ------------------------------------------------------------------------------------------------------

# Importing libraries
# ------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import tqdm as tqdm
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------------------------------------------------------------------------------------

# Auxiliary functions
# ------------------------------------------------------------------------------------------------------
# Return RHS of SIR ODE system
def SIR(x, t, N, beta, gamma):
    S, I, R = x
    xdot = [
        -(beta*S*I)/N,
        (beta*S*I)/N - gamma*I,
        gamma*I
    ]
    return xdot

# Generating observational data
def get_obs_data(t, S,I,R):
    # Generating observations
    # Generate 10 equispaced time locations in the domain for fair comparison with ABC-SMC
    obs_ind = np.linspace(0, len(t) - 1, 10, dtype=int)
    S_obs = np.array([S[ind] for ind in obs_ind])
    I_obs = np.array([I[ind] for ind in obs_ind])
    R_obs = np.array([R[ind] for ind in obs_ind])
    # Generate Gaussian noise for S_obs, I_obs, and R_obs
    noise_S = 0.01 * S_obs * np.random.randn(len(S_obs))
    noise_I = 0.01 * I_obs * np.random.randn(len(I_obs))
    noise_R = 0.01 * R_obs * np.random.randn(len(R_obs))
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

    return t_obs, u_obs_S, u_obs_I, u_obs_R

# Simulation block, training the PINN
def simulate(t_obs, u_obs_S, u_obs_I, u_obs_R, t_physics, N, target_params, tol, max_its, lambda_weight):
    # Training setup
    # Initialize PINN
    pinn = FCN(1,1,32,3)
    # Sample initial problems parameters from uniform 0 to 10 for comparison to be fair
    beta = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    gamma = torch.nn.Parameter(torch.empty(1).uniform_(0, 10), requires_grad=True)
    # Define optimiser and make problem parameters learnable
    optimiser = torch.optim.Adam(list(pinn.parameters())+[beta,gamma],lr=1e-2)
    beta_t, gamma_t = target_params[0], target_params[1] # Target params
    betas, gammas = [], [] # To keep track of progress
    beta_times, gamma_times = [], [] # To keep track of execution time at each iteration (for plotting purposes)
    beta_done, gamma_done = False, False # Keep track of state of each parameter individually

    # Start timing
    start_time = time.time()

    for it in range(max_its):
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
        # Record values, stop recording invidually if one of them is done
        if not beta_done:
            curr_time = time.time()
            beta_times.append(curr_time-start_time)
            betas.append(beta.item())
            beta_done = True if abs(beta.item() - beta_t) < tol else False
        if not gamma_done:
            curr_time = time.time()
            gamma_times.append(curr_time - start_time)
            gammas.append(gamma.item())
            gamma_done = True if abs(gamma.item() - gamma_t) < tol else False
        if beta_done and gamma_done:
            end_time = time.time()
            final_time = end_time - start_time
            # returns: betas, gammas, it, stopped, total_exec_time, beta_times, gamma_times
            return betas, gammas, it, True, final_time, beta_times, gamma_times
    
    end_time = time.time()  # End timing after loop if no early stop
    final_time = end_time - start_time
    # returns: betas, gammas, it, stopped, total_exec_time, beta_times, gamma_times
    return betas, gammas, it, False, final_time, beta_times, gamma_times

# Generate basic statistical report with plots and summary statistics
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
    plt.yticks([])  # Hides y-axis labels
    plt.xlabel("Execution Time (s)")
    plt.savefig('./sim_results/execution_time_pinn.png')
# ------------------------------------------------------------------------------------------------------

# Defining the network
# ------------------------------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------------------------------

if __name__=="__main__": 
    #                                    PROBLEM SETUP
    # --------------------------------------------------------------------------------------------------
    # Defining the problem
    N = 100  # Total population
    # Initial number of infected and recovered individuals
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially
    S0 = N - I0 - R0
    # Initial state of the system
    X0 = [S0, I0, R0]
    # Target parameters
    beta, gamma = 1.5, 0.5
    # A grid of time points (in days)
    finalT = 17.0
    t  = np.linspace(0, finalT, 100)
    # Solving the problem numerically (via odeint numerical solver)
    result = odeint(SIR, X0, t, args=(N, beta, gamma))
    S, I, R = result.T
    # Get observational data (noisy)
    t_obs, u_obs_S, u_obs_I, u_obs_R = get_obs_data(t, S,I,R)
    # Tensor of times to train PINN through physics loss
    num_phys_locs = 750
    t_physics = torch.linspace(0, finalT, num_phys_locs).view(-1, 1).requires_grad_(True)
    # Hyperparameter deciding weight given to fitting observational data
    lambda_weight = 10
    # Simulation setup
    num_sim = 100 # Number of simulations to run
    tol = 0.1 # Tolerance to target needed to stop training
    max_its = 10000
    # --------------------------------------------------------------------------------------------------

    #                                    SIMULATION LOOP
    # --------------------------------------------------------------------------------------------------
    # Set up a figure with two subplots: one for betas and one for gammas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12))
    fig.suptitle("Parameter trajectories for all simulations")
    total_exec_times, last_beta_times, last_gamma_times = [], [], []

    for sim_id in tqdm.tqdm(range(1,num_sim+1)):
        betas, gammas, it, stopped, total_exec_time, beta_times, gamma_times = simulate(t_obs, u_obs_S, u_obs_I, u_obs_R, t_physics, N, [beta,gamma], tol, max_its, lambda_weight)
        total_exec_times.append(total_exec_time)
        last_beta_times.append(beta_times[-1]) # Keep track of last beta time for plot
        last_gamma_times.append(gamma_times[-1]) 
        # Plot the beta trajectory for this simulation on the first subplot
        ax1.plot(beta_times, betas)
        # Plot the gamma trajectory for this simulation on the second subplot
        ax2.plot(gamma_times, gammas)
    # --------------------------------------------------------------------------------------------------

    #                                    PLOTTING PARAM TRAJ
    # --------------------------------------------------------------------------------------------------
    # Configure the first subplot (betas)
    ax1.set_title("Beta values")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Beta value")
    ax1.set_xlim(0, max(last_beta_times))
    ax1.set_ylim(-2,12)
    ax1.axhline(y=beta + tol, color='tab:blue', linestyle='dashed', linewidth=1, label=f'beta + tol')
    ax1.axhline(y=beta - tol, color='tab:red', linestyle='dashed', linewidth=1, label=f'beta - tol')
    ax1.fill_between([0, max(last_beta_times)], beta - tol, beta + tol, color='green', alpha=0.3, label='Convergence region')
    ax1.legend(loc="best")
    # Configure the second subplot (gammas)
    ax2.set_title("Gamma values")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Gamma value")
    ax2.set_xlim(0, max(last_gamma_times))
    ax2.set_ylim(-2,12)
    ax2.axhline(y=gamma + tol, color='tab:blue', linestyle='dashed', linewidth=1, label=f'gamma + tol')
    ax2.axhline(y=gamma - tol, color='tab:red', linestyle='dashed', linewidth=1, label=f'gamma - tol')
    ax2.fill_between([0, max(last_gamma_times)], gamma - tol, gamma + tol, color='green', alpha=0.3, label='Convergence region')
    ax2.legend(loc="best")
    # Show the plot
    plt.tight_layout()
    plt.savefig('./sim_results/param_trajec_pinn.png')
    # --------------------------------------------------------------------------------------------------
    
    # Generate statistical report on execution times
    stat_report(total_exec_times)