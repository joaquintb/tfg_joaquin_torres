# Physics-Informed Neural Networks for Parameter Inversion in Dynamical Systems

This repository contains the implementation and results of my bachelor thesis, which focuses on the application of Physics-Informed Neural Networks (PINNs) and Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC) algorithms for parameter inversion in dynamical systems. Specifically, it includes solutions and analyses for the Lotka-Volterra (LV) and Susceptible-Infected-Recovered (SIR) models.

## Repository Contents

### Lotka-Volterra (folder)

- **sim_results**: Contains statistical analyses of the execution times, including plots and summary statistics.
- **abc_smc_lv_script.py**: Python script to run ABC-SMC algorithm simulations for parameter inference in the LV model.
- **pinn_lv_script.py**: Python script to run simulations using PINNs for parameter inference in the LV model.
- **pinn_lv.ipynb**: Jupyter notebook used as a draft for generating plots and testing various implementations for the LV model.

### SIR (folder)

- **sim_results**: Contains statistical analyses of the execution times, including plots and summary statistics.
- **abc_smc_sir_script.py**: Python script to run ABC-SMC algorithm simulations for parameter inference in the SIR model.
- **pinn_sir_script.py**: Python script to run simulations using PINNs for parameter inference in the SIR model.
- **pinn_sir.ipynb**: Jupyter notebook used as a draft for generating plots and testing various implementations for the SIR model.

### Reference Materials

- **[3]_bayesian_inference_complex_network.pdf**: Paper by Vanessa García López-Mingo, which serves as one of the main references for the methodologies implemented in this project.

## License

This project is open-sourced under the MIT License.

## Acknowledgments

Portions of this code are adapted from other works. Please refer to individual files for specific attributions.
