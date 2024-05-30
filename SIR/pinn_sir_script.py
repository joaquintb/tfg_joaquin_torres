# Author: Joaquín Torres Bravo
# Date: June 2024
# Bachelor Thesis

# Libraries
import numpy as np
import torch
import torch.nn as nn
import tqdm as tqdm
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def get_obs_data(S,I,R):
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

# Function to simulate training the PINN
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
        # Add current parameter values to lists
        betas.append(beta.item())
        gammas.append(gamma.item())
        # Check if parameter distance within tolerance
        if abs(beta.item() - beta_t) < tol and abs(gamma.item() - gamma_t) < tol:
            end_time = time.time()  # End timing here if early stop
            return betas, gammas, it, True, end_time - start_time  # Return elapsed time
    end_time = time.time()  # End timing after loop if no early stop
    return betas, gammas, it, False, end_time - start_time

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

    plt.savefig('./SIR/sim_results/executon_time_pinn.png')

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
    # Get observational data (noisy)
    t_obs, u_obs_S, u_obs_I, u_obs_R = get_obs_data(S,I,R)
    # Tensor of times to train PINN through physics loss
    num_phys_locs = 750
    t_physics = torch.linspace(0, finalT, num_phys_locs).view(-1, 1).requires_grad_(True)
    # Hyperparameter deciding weight given to fitting observational data
    lambda_weight = 10
    # Simulation setup
    num_sim = 100 # Number of simulations to run
    tol = 0.1 # Tolerance to target needed to stop training
    max_its = 10000

    largest_it = -1
    # Set up a figure with two subplots: one for betas and one for gammas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12))
    # Execution times vectors
    exec_times = []
    fig.suptitle("Parameter trajectories for all simulations")
    for sim_id in range(1,num_sim+1):
        betas, gammas, it, stopped, exec_time = simulate(t_obs, u_obs_S, u_obs_I, u_obs_R, t_physics, N, [beta,gamma], tol, max_its, lambda_weight)
        exec_times.append(exec_time)
        largest_it = it if it > largest_it else largest_it
        # Plot the beta trajectory for this simulation on the first subplot
        ax1.plot(betas)
        # Plot the gamma trajectory for this simulation on the second subplot
        ax2.plot(gammas)
        # print(f'SIM {sim_id}: beta0={betas[0]}, gamma0={gammas[0]} betaF={betas[-1]}, gammaF={gammas[-1]}, t={exec_time} seconds, stop={stopped}')

    # Configure the first subplot (betas)
    ax1.set_title("Beta values")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Beta value")
    ax1.set_xlim(0, largest_it)
    ax1.set_ylim(-2,12)
    ax1.hlines(beta, 0, largest_it, color="tab:green", linestyles='dashed', label="True Beta value", lw=6)
    ax1.legend(loc="best")
    # Configure the second subplot (gammas)
    ax2.set_title("Gamma values")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Gamma value")
    ax2.set_xlim(0, largest_it)
    ax2.set_ylim(-2,12)
    ax2.hlines(gamma, 0, largest_it, color="tab:green", linestyles='dashed', label="True Gamma value", lw=6)
    ax2.legend(loc="best")
    # Show the plot
    plt.savefig('./SIR/sim_results/param_trajec_pinn.png')

    stat_report(exec_times)