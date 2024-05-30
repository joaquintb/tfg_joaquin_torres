import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from scipy.optimize import minimize
from scipy.special import logsumexp
import sys,ast
from random import choices,seed,random
from functools import partial
import os
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
from scipy.integrate import odeint
import math
import time
import seaborn as sns

# SIR, no latent period, but with reocevered becoming suspitible again
def deriv_SIR(y, t, N, betaI, gammaI):
    beta = betaI
    gamma = gammaI
    S, I, R = y
    Nt = S + I + R
    dSdt = -beta * S * I / Nt 
    dIdt = beta * S * I / Nt - gamma * I 
    dRdt = gamma * I 
    return [dSdt, dIdt, dRdt]

def euc_dist(data1, data2):
    if np.shape(data1) != np.shape(data2):
        print ("\n the dimensions of the datasets are different (%s v.s. %s)\n" % (len(data1), len(data2)))
        sys.exit()
    else:
        z =np.array((data1[:,0] - data2[:,0])**2+ (data1[:,1] - data2[:,1])**2)
        #print (z)
        distance=np.sum(z)

    if distance < 0:
        return [None]
    else:
        return distance

def euc_disti(data1, data2):
    if np.shape(data1) != np.shape(data2):
        print ("\n the dimensions of the datasets are different (%s v.s. %s)\n" % (len(data1), len(data2)))
        sys.exit()
    else:
        #data2=data2.T
        distance =np.linalg.norm(data1[1]-data2[1])+np.linalg.norm(data1[2]-data2[2])
    if distance < 0:
        return [None]
    else:
        return distance
    
def prior():
### Generate a random parameter inside the limits stablished. The shape of the distribution can be changed if required
    prior = []
    for ipar,par in enumerate(params_SIR):
        prior.append(uniform.rvs(loc = par['lower_limit'],
                                 scale = par['upper_limit'])) #par['upper_limit'])) #par['upper_limit']))
        
       
    return prior

#function that given the values of the parameters, calculates the 

def evaluate_prev_pru(params):
    #print('parameters',params)
    l=len(params)
    prior = 1
    for ipar,par in enumerate(params_SIR):
    #for i in range(l):
        prior *= uniform.pdf(params[ipar],loc = par['lower_limit'],
                                 scale = par['upper_limit'])
        if prior==0:
            break   
      #  print('params i', params[i])
       # print('prior',prior)
    return prior

#function that, given a list of parameters sampled, perturbs it by applying a multivariate normal kernel
def perturb(listaprev,s):
    #print(listaprev)
    lista=np.asarray(listaprev) #.tolist()
    #mean_vec=np.mean(lista)
    cov_matrix=2.0*np.cov(lista.T)  #the covariance matrix for the multivariate normal perturbation kernel is given by this expression
    kernel=multivariate_normal(cov=cov_matrix)
    pert=s+kernel.rvs() # here we obtain the list of perturbed parameters
    pertur=pert.tolist()
    return pertur

def rk4(f,in_c,t,params):
    #params=[a,b,c,d]
    h=t[1]-t[0]
    n=len(t)
    X  = np.zeros([n,len(in_c)])
    X[0]=in_c
    for i in range(n-1):
        #print(params)
        #print(X[i])
        k1=f(X[i],t[i],N,*params)
        #print(k1)
        k2=f(X[i]+k1*h/2.,t[i]+h/2.,N,*params)
        k3=f(X[i]+k2*h/2.,t[i]+h/2,N,*params)
        k4=f(X[i]+k3*h,t[i]+h,N,*params)
    
        X[i+1]=X[i]+h*(k1/6.+k2/3.+k3/3.+k4/6.)
     
    return X

#function that gives the denominator used to calculate the weights of every particle.
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

def principal(epsilons,listaparametros,N,data1, tol_target):
   # accepted_distances = np.loadtxt('smc/distances_{}_{}_{}_{}.out'.format(model,sto,gamma,prior_label))
    T=len(epsilons)
    weight=np.zeros((T,N),float)
    dist=np.zeros((T,N),float)
    sample=np.zeros((T,N),list)
    nT=100
    X0=[S0,I0,R0]
    t = np.linspace(0, finalT, 10)
    #t=np.linspace(0.,10,10)

    start_time = time.time()

    for i in range(T):
        count=0
        counti=0
        #print("SMC step with target distance: {}".format(epsilons[i]))
        if i==0:
            for j in range (N):
                dist[i,j]=epsilons[i]+1
                while dist[i,j]>epsilons[i]:
                    sample[i,j]=prior()
                    #print(sample[i,j])
                    #sample[i,j]=np.array(prior())
                    sample[i,j]=np.asarray(sample[i,j])
                    data2= odeint(deriv_SIR, X0, t,args=(nT,*sample[i,j]))
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
                        data2=odeint(deriv_SIR, X0, t,args=(nT, *sample[i,j]))
                        data2=np.array(data2)
                        #print('data2',data2)
                        dist[i,j]=euc_disti(data1,data2)
                        #print('distendata2',dist[i,j])
                    counti+=1
                    #print(counti)
        
        for j in range(N):
            if i==0:
                weight[i,j]=1
               # #print(weight[i,j])
            else:
                denom=weighting(i,j,N,sample[i,:],weight[i-1,:],list(sample[i-1,:]))
                weight[i,j]=evaluate_prev_pru(sample[i,j])/denom
        ##print('weight[i,:]',weight[i,:])
        if i!=0:
           weight[i,:]=normalize(weight[i,:])
           #print('weight[i,:] normalized',weight[i,:])
        

        # Check convergence using tolerance targets for beta and gamma
        best_index = np.argmin(dist[i, :])
        best_params = sample[i, best_index]
        if all(abs(best_params[k] - listaparametros[k]['target_value']) < tol_target[k] for k in range(len(best_params))):
            #print(f"Converged to within tolerance at iteration {i + 1}.")
            #print(f"Best parameters: Beta: {best_params[0]}, Gamma: {best_params[1]}")
            end_time = time.time()
            return sample, weight, dist, data2, True, end_time - start_time  # Return with a flag indicating early stopping
    
        #pars = np.loadtxt('smc_van/pars_{}.out'.format(i))
        #weights = np.loadtxt('smc_van/weights_{}.out'.format(i))
        #np.savetxt('smc_van/pars_{}.out'.format(i), sample[T-1,:])
        #np.savetxt('smc_van/weights_{}.out'.format(i), weight[T-1,:])
      #  np.savetxt('smc/distances_{}.out'.format(label), accepted_distances)
    ##print('sample',sample[T-1,N-1])
    ##print('weight',weight[T-1])
    #print('dist',dist[T-1])
    end_time = time.time()
    return sample, weight, dist,data2, False, end_time - start_time

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

    plt.savefig('./SIR/sim_results/executon_time_abc.png')

if __name__ == "__main__":
    # Total population, N.
    N = 100
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    X0=[S0,I0,R0]
    betaI, gammaI = 1.5, 0.5
    # A grid of time points (in days)
    finalT = 17.0
    t  = np.linspace(0, finalT, 100)
    t2 = np.linspace(0, finalT, 10)
    # Initial conditions vector
    y0 = S0, I0, R0
    # Observations
    SIR = scipy.integrate.odeint(deriv_SIR, y0, t2, args=(N, betaI, gammaI))
    S, I, R = SIR.T  # Transpose to separate S, I, R
    # Generate Gaussian noise
    noise_S = 0.01 * S * np.random.randn(len(S))
    noise_I = 0.01 * I * np.random.randn(len(I))
    noise_R = 0.01 * R * np.random.randn(len(R))
    # Add noise to observations
    S_obs = S + noise_S
    I_obs = I + noise_I
    R_obs = R + noise_R
    # Combine the noisy observations into a single array for ABC-SMC
    SIR = np.vstack((S_obs, I_obs, R_obs)).T  # Transpose to get columns as S, I, R

    epsilons=[60, 50, 40, 20, 10, 5,2,1.5,1,0.5]

    params_SIR = [
    {'name': 'beta', 'lower_limit': 0, 'upper_limit': 10.0, 'target_value': betaI},
    {'name': 'gamma', 'lower_limit': 0, 'upper_limit': 10.0, 'target_value': gammaI}
    ]

    # Define tolerance distance to target parameters for early stopping
    tol_target = [0.1, 0.1]

    num_sim = 100
    exec_times = []

    for sim_id in range(1, num_sim+1):
        print(f'SIM {sim_id}')
        sample,weight,dist,data2, stopped_early, exec_time =principal(epsilons,params_SIR,100,SIR, tol_target) 
        exec_times.append(exec_time)

    stat_report(exec_times)