import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.optimize import minimize
from scipy.special import logsumexp
import sys, ast
from random import choices, seed, random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Define the tolerances for each iteration
epsilons = [30.0, 16.0, 6.0, 5.0, 4.3]

# Define the parameter ranges for the Lotka-Volterra model
params_lotka_volterra = [
    {'name': 'a', 'lower_limit': 0.0, 'upper_limit': 10.0},  # growth rate of prey in absence of predators
    {'name': 'b', 'lower_limit': 0.0, 'upper_limit': 10.0},  # predation rate
    {'name': 'c', 'lower_limit': 0.0, 'upper_limit': 10.0},  # mortality rate of predators
    {'name': 'd', 'lower_limit': 0.0, 'upper_limit': 10.0}   # rate at which predators increase by consuming prey
]

# Calculate the Euclidean distance between two datasets
def euc_dist(data1, data2):
    if np.shape(data1) != np.shape(data2):
        print("\n The dimensions of the datasets are different (%s v.s. %s)\n" % (len(data1), len(data2)))
        sys.exit()
    else:
        distance = np.linalg.norm(data1 - data2)
        print('dist', data1 - data2)
    return distance if distance >= 0 else None

# Calculate the Euclidean distance for two-dimensional datasets
def euc_disti(data1, data2):
    if np.shape(data1) != np.shape(data2):
        print("\n The dimensions of the datasets are different (%s v.s. %s)\n" % (len(data1), len(data2)))
        sys.exit()
    else:
        z = np.array((data1[:, 0] - data2[:, 0]) ** 2 + (data1[:, 1] - data2[:, 1]) ** 2)
        distance = np.sum(z)
    return distance if distance >= 0 else 100000 if np.isnan(distance) else None

# Generate a random parameter inside the limits established
def prior():
    prior = []
    for par in params_lotka_volterra:
        prior.append(uniform.rvs(loc=par['lower_limit'], scale=par['upper_limit'] - par['lower_limit']))
    return prior

# Define the Lotka-Volterra model
def lotka_volterra(X, t, params):
    a, b, c, d = params
    dxdt = a * X[0] - b * X[0] * X[1]
    dydt = d * X[0] * X[1] - c * X[1]
    return np.array([dxdt, dydt])

# Runge-Kutta 4th order method for solving differential equations
def rk4(model, X0, t, params):
    X = np.zeros((len(t), len(X0)))
    X[0] = X0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = dt * model(X[i - 1], t[i - 1], params)
        k2 = dt * model(X[i - 1] + 0.5 * k1, t[i - 1] + 0.5 * dt, params)
        k3 = dt * model(X[i - 1] + 0.5 * k2, t[i - 1] + 0.5 * dt, params)
        k4 = dt * model(X[i - 1] + k3, t[i - 1] + dt, params)
        X[i] = X[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X

# Perturb the parameters (assuming a simple Gaussian perturbation)
def perturb(params):
    perturbed_params = []
    for par in params:
        perturbed_params.append(norm.rvs(loc=par, scale=0.1))
    return np.clip(perturbed_params, 0, 10)  # Ensure parameters stay within bounds

# Main ABC-SMC algorithm
def abc_smc(num_particles, num_iterations, epsilons):
    particles = []
    distances = []
    
    # Initial population from the prior distribution
    for _ in range(num_particles):
        params = prior()
        X0 = [10, 5]
        t = np.linspace(0, 20, 100)
        data = rk4(lotka_volterra, X0, t, params)
        dist = euc_disti(data, observed_data)  # Assuming observed_data is defined
        particles.append(params)
        distances.append(dist)
    
    for i in range(1, num_iterations):
        epsilon = epsilons[i]
        new_particles = []
        new_distances = []
        
        for _ in range(num_particles):
            idx = np.argmin(distances)  # Select the particle with the minimum distance
            params = perturb(particles[idx])  
            X0 = [10, 5]
            t = np.linspace(0, 20, 100)
            data = rk4(lotka_volterra, X0, t, params)
            dist = euc_disti(data, observed_data)
            
            if dist <= epsilon:
                new_particles.append(params)
                new_distances.append(dist)
        
        particles = new_particles
        distances = new_distances
    
    return particles, distances

if __name__ == "__main__":
    # Run the ABC-SMC algorithm
    num_particles = 100  # Example number of particles
    num_iterations = len(epsilons)
    observed_data = np.random.rand(100, 2)  # Example observed data, replace with actual data
    particles, distances = abc_smc(num_particles, num_iterations, epsilons)

    # Extract and analyze the final population
    final_population = np.array(particles)
    params_mean = np.mean(final_population, axis=0)
    params_median = np.median(final_population, axis=0)
    params_var = np.var(final_population, axis=0)

    print("Mean of parameters:", params_mean)
    print("Median of parameters:", params_median)
    print("Variance of parameters:", params_var)

    # Plotting parameter distributions
    for i, param in enumerate(['a', 'b', 'c', 'd']):
        plt.figure()
        plt.hist(final_population[:, i], bins=30, edgecolor='black')
        plt.title(f'Distribution of Parameter {param}')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.show()

    # Plotting model output vs observed data
    t = np.linspace(0, 20, 100)
    plt.figure()
    plt.plot(t, observed_data[:, 0], 'o', label='Observed Prey')
    plt.plot(t, observed_data[:, 1], 'o', label='Observed Predator')

    for params in final_population:
        data = rk4(lotka_volterra, [10, 5], t, params)
        plt.plot(t, data[:, 0], label='Simulated Prey', alpha=0.3)
        plt.plot(t, data[:, 1], label='Simulated Predator', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()