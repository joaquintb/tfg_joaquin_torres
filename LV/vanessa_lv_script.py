import numpy as np
from scipy.stats import uniform, multivariate_normal
import sys
from random import choices, seed
from tqdm import tqdm
import matplotlib.pyplot as plt
import time as time
import seaborn as sns

# Calculate the Euclidean distance between two datasets
def euc_dist(data1, data2):
    if np.shape(data1) != np.shape(data2):
        print ("\n the dimensions of the datasets are different (%s v.s. %s)\n" % (len(data1), len(data2)))
        sys.exit()
    else:
        distance = np.linalg.norm(data1 - data2)
        print('dist',data1 - data2)
    if distance < 0:
        return [None]
    else:
        return distance

# Calculate the Euclidean distance for two-dimensional datasets
def euc_disti(data1, data2):
    if np.shape(data1) != np.shape(data2):
        print ("\n the dimensions of the datasets are different (%s v.s. %s)\n" % (len(data1), len(data2)))
        sys.exit()
    else:
        z =np.array((data1[:,0] - data2[:,0])**2+ (data1[:,1] - data2[:,1])**2)
        distance=np.sum(z)
    if distance < 0:
        return [None]
    elif np.isnan(distance):
      distance=100000
      return distance
    else:
        return distance

def prior():
### Generate a random parameter inside the limits stablished. The shape of the distribution can be changed if required
    prior = []
    for ipar,par in enumerate(params_lotka_volterra):
        prior.append(uniform.rvs(loc = par['lower_limit'],
                                 scale = par['upper_limit'])) #par['upper_limit']))
    return prior

#function that given the values of the parameters, calculates the 

def evaluate_prev_pru(params):
    #print('parameters',params)
    l=len(params)
    prior = 1
    for ipar,par in enumerate(params_lotka_volterra):
    #for i in range(l):
        prior *= uniform.pdf(params[ipar],loc = par['lower_limit'],
                                 scale = par['upper_limit'])
        if prior==0:
            break   
    return prior

#function that, given a list of parameters sampled, perturbs it by applying a multivariate normal kernel
def perturbi(listaprev,s):
    #print(listaprev)
    lista=np.asarray(listaprev) #.tolist()
    #mean_vec=np.mean(lista)
    k=uniform.rvs(loc = -0.1,scale = 0.2)
    cov_matrix=2.0*np.cov(lista.T)  #the covariance matrix for the multivariate normal perturbation kernel is given by this expression
    kernel=multivariate_normal(cov=cov_matrix)
    pert=s+k # here we obtain the list of perturbed parameters
    pertur=pert.tolist()
    return pertur

def perturb(listaprev,s):
    #print(listaprev)
    lista=np.asarray(listaprev) #.tolist()
    #mean_vec=np.mean(lista)
    cov_matrix=2.0*np.cov(lista.T)  #the covariance matrix for the multivariate normal perturbation kernel is given by this expression
    kernel=multivariate_normal(cov=cov_matrix)
    pert=s+kernel.rvs() # here we obtain the list of perturbed parameters
    pertur=pert.tolist()
    return pertur

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

# Define the Lotka-Volterra model
def lotka_volterra(X, t, params):
    a, b, c, d = params
    dxdt = a * X[0] - b * X[0] * X[1]
    dydt = d * X[0] * X[1] - c * X[1]
    return np.array([dxdt, dydt])

def weighting(i,j,N,sam,wei,sampre):
     denom=0
     #ker=1
     samprev=np.asarray(sampre)
     cov_matrix=2.0*np.cov(samprev.T)
     kernel=multivariate_normal(cov=cov_matrix)
     for k in range(N):
            #print('sample i j',type(sam[k]),sam[k])
           # print('sample i-1,j',type(sampre[k]),sampre[k])
            sampre[k]=np.array(sampre[k])
            #print('sampre',sampre[k])
            #cov_matrix=2.0*np.cov((sampre[k]).T)  #the covariance matrix for the multivariate normal perturbation kernel is given by this expression
            #print('cov',cov_matrix)
            #kernel=multivariate_normal(cov=cov_matrix)
            # print('wei',wei[i-1,k])
            #print('sam[j]',sam[j])
            #print('sampre[k]',sampre[k])
            ker=kernel.pdf(sam[j]-sampre[k])
            #print('ker',ker)
            #kerne=np.prod(ker)  #here we are obtaining the joint probability of the parameter vector obtained when applying the kernel
            denom+=wei[k]*ker #kerne
            #print('kernel',kernel.cdf(sam[k]-sampre[k]))
     #print('den',denom)      
     return denom

#function used to normalize the weights
def normalize(wei):
    #normalized=wei/np.linalg.norm(wei)
    normalized=wei/np.sum(wei)
    return normalized  

def principal(epsilons,listaparametros,N,data1,t, tol_target):
   # accepted_distances = np.loadtxt('smc/distances_{}_{}_{}_{}.out'.format(model,sto,gamma,prior_label))
    T=len(epsilons)
    weight=np.zeros((T,N),float)
    dist=np.zeros((T,N),float)
    sample=np.zeros((T,N),list)
    X0=[0.5,1]
    #t=np.linspace(0.,10,10)
    for i in range(T):
        count=0
        counti=0
        label=i
        #print("SMC step with target distance: {}".format(epsilons[i]))
        if i==0:
            for j in range (N):
                dist[i,j]=epsilons[i]+1
                while dist[i,j]>epsilons[i]:
                    sample[i,j]=prior()
                    #sample[i,j]=np.array(prior())
                    sample[i,j]=np.asarray(sample[i,j])
                    data2= rk4(lotka_volterra,X0,t,sample[i,j])
                    #print('data2',data2)
                    #data2=np.array(data2, dtype=np.float64)
                    dist[i,j]=euc_disti(data1,data2)
                    #print('distcondata2',dist[i,j])
                count+=1
                #print(count)
       
        else:
            for j in range (N):
                dist[i,j]=epsilons[i]+1
                while dist[i,j]>epsilons[i]:
                    seed()
                    np.random.seed()
                    choose = choices(sample[i-1,:], weights = weight[i-1,:],k=1)[0] # select a point from the previous sample
                    sample[i,j]=choose
                    #print("before perturb",type(sample[i,j]))
                    #print("before perturb",list(sample[i-1,:]))
                    sample[i,j] = perturb(list(sample[i-1,:]),sample[i,j]) # and perturb it
                    #print("after perturb", sample[i,j])
                    #print("after perturb", type(sample[i,j]))
                    evaluation=evaluate_prev_pru(sample[i,j]) 
                    if evaluation>0:
                        data2=rk4(lotka_volterra,X0,t,sample[i,j])
                        data2=np.array(data2)
                        #print('data2',data2)
                        dist[i,j]=euc_disti(data1,data2)
                        #print('distendata2',dist[i,j])
                counti+=1
                #print(counti)
        for j in range(N):
            if i==0:
                weight[i,j]=1
               # print(weight[i,j])
            else:
                denom=weighting(i,j,N,sample[i,:],weight[i-1,:],list(sample[i-1,:]))
                weight[i,j]=evaluate_prev_pru(sample[i,j])/denom
        #print('weight[i,:]',weight[i,:])
        if i!=0:
           weight[i,:]=normalize(weight[i,:])

        # Check convergence using tolerance targets for beta and gamma
        best_index = np.argmin(dist[i, :])
        best_params = sample[i, best_index]
        if all(abs(best_params[k] - listaparametros[k]['target_value']) < tol_target[k] for k in range(len(best_params))):
            print(f"Converged to within tolerance at iteration {i + 1}.")
            print(f"Best parameters: a={best_params[0]}, b={best_params[1]}, c={best_params[2]}, d={best_params[3]}")
            return sample, weight, dist, data2, True  # Return with a flag indicating early stopping
           #print('weight[i,:] normalized',weight[i,:])
        #pars = np.loadtxt('smc_van/pars_{}.out'.format(i))
        #weights = np.loadtxt('smc_van/weights_{}.out'.format(i))
        #np.savetxt('smc_van/pars_{}.out'.format(i), sample[T-1,:])
        #np.savetxt('smc_van/weights_{}.out'.format(i), weight[T-1,:])
      #  np.savetxt('smc/distances_{}.out'.format(label), accepted_distances)
    #print('sample',sample[T-1,N-1])
    #print('weight',weight[T-1])
    #print('dist',dist[T-1])
    return sample, weight, dist,data2, False


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

    plt.savefig('./sim_results/execution_time_abc.png')

if __name__ == "__main__":

    # Define the tolerances for each iteration
    epsilons=[30.0,16.0,6.0,4.3,3.5,2.1,1.2,0.8,0.2,0.08]
    parametros = [1,1,1,1]
    # Define the parameter ranges for the Lotka-Volterra model
    params_lotka_volterra = [
        {'name': 'a', 'lower_limit': 0.0, 'upper_limit': 5.0, 'target_value': parametros[0]},  # growth rate of prey in absence of predators
        {'name': 'b', 'lower_limit': 0.0, 'upper_limit': 5.0, 'target_value': parametros[1]},  # predation rate
        {'name': 'c', 'lower_limit': 0.0, 'upper_limit': 5.0, 'target_value': parametros[2]},  # mortality rate of predators
        {'name': 'd', 'lower_limit': 0.0, 'upper_limit': 5.0, 'target_value': parametros[3]}   # rate at which predators increase by consuming prey
    ]

    X0=[0.5,1]
    # 8 equispaced time locations
    t=[1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4]
    # To solve system numerically
    t1=np.linspace(0,15,1000)
    # Solution at 8 observational points
    midata=rk4(lotka_volterra,X0,t,parametros)
    x,y = midata.T    
    # Generate Gaussian Noise
    noise_x = 0.01 * x * np.random.randn(len(x))
    noise_y = 0.01 * y * np.random.randn(len(y))
    x_obs = x + noise_x
    y_obs = y + noise_y
    midata = np.vstack((x_obs, y_obs)).T

    # Define tolerance distance to target parameters for early stopping
    tol_target = [0.1, 0.1, 0.1, 0.1]

    num_sim = 100
    exec_times = []
    for sim_id in tqdm(range(1, num_sim+1)):
        print(f'SIM {sim_id}')
        start_time = time.time()
        sample,weight,dist,data2, stopped_early =principal(epsilons,params_lotka_volterra,100,midata,t, tol_target)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Total time for current simulation: {total_time}')
        exec_times.append(total_time)

    stat_report(exec_times)