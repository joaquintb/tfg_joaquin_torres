import numpy as np

# Parameters and initial conditions
t0 = 0
tf = 15
num_points = 1000
dt = (tf - t0) / (num_points - 1)
t_generated = np.linspace(t0, tf, num_points)

print(f"Calculated dt: {dt}")

# Given specific times to check
t_check = np.array([1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4])

# Function to check if each time in t_check is in t_generated within a tolerance
def check_times(t_generated, t_check, tol=1e-3):
    return np.isclose(t_check[:, None], t_generated, atol=tol).any(axis=1)

# Check inclusion
included = check_times(t_generated, t_check)

for time, inc in zip(t_check, included):
    print(f"Time {time} is in the generated vector: {inc}")
