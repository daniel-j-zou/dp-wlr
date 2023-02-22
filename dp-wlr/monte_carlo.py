# This module includes all functions related to Monte-Carlo, both in generating data, 
# summarizing data, and plotting it

import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, pstdev
from run_wlr import private_wlr, non_private_wlr
import os

# This function does data generation for Monte Carlo
# Inputs: ns - array of sample sizes, ks - array of chosen ks, epsilons - array of epsilons,
# tests - array of 0,1,2 where
# 0 is private and weighted, 1 is private and non-weighted, 2 is non-private and weighted
# intercept_eps - array of intercept_eps
# iterations is the number of times each of these is run
# data_function is a function that generates the data given n
# intercept - boolean on whether intercept is generated
# lower and upper bounds for exponential
#
# Outputs: 
# Saves as a dataframe
# Each row is one iteration with the columns being
# n, k, epsilon, test_index, intercept_epsilon, slope, inter
 
def data_gen(ns, ks, epsilons, tests, intercept_eps, iterations, data_function, 
    intercept = False, lower_bound = -50, upper_bound = 50):

    df = np.array(np.meshgrid(ns, ks, epsilons, tests, intercept_eps, [0], [0])).T.reshape(-1,7)
    df = np.repeat(df, repeats = iterations, axis = 0)

    for i in range(np.shape(df)[0]):
        params = df[i]
        params = [int(x) for x in params]
        dfi = data_function(params[0])

        # private and weighted

        if params[3] == 0:
            df[i][5] = private_wlr(dfi, params[1], params[0], params[2], True, 
                intercept, params[4], lower_bound, upper_bound)
        
        # private and non-weighted

        if params[3] == 1: 
            df[i][5] = private_wlr(dfi, params[1], params[0], params[2], False, 
                intercept, params[4], lower_bound, upper_bound)

        # non-private and weighted

        if params[3] == 2:
            df[i][5] = non_private_wlr(dfi, params[1], params[0], True, intercept,
            lower_bound, upper_bound)

    df = np.around(df, 3)
    name = 'data' + str(ns) + str(ks) + str(epsilons)+ str(tests) +str(intercept_eps)+ str(iterations)
    np.save(os.path.join('data', name), df)
    return(df)


# Computes summary statistics from datagen
# Inputs: df - data frame or name of data frame, iterations, summary function, intercepts or no 
# 
# Outputs: new array with rows representing different configurations
# columns are: n, k, eps, test_type, inter_eps, slope_summary, inter_summary (0 if none)

def summary_statistics(df, iterations, fn, intercept = False):
    if isinstance(df, str):
        df = np.load(df)

    n = np.shape(df)[0]
    diff = int(n / iterations)
    newdf = df[[i for i in range(0, n, iterations)],:]
    df = np.transpose(df)
    for i in range(diff):
        slope_data = df[5][i*iterations : (i+1) * iterations - 1]
        slope_summary = fn(slope_data)
        newdf[i][5] = slope_summary
    if intercept:
        for i in range(diff):
            inter_data = df[6][i*iterations : (i+1) * iterations - 1]
            inter_summary = fn(inter_data)
            newdf[i][6] = inter_summary
    
    newdf = np.around(newdf, 3)
    
    return(newdf)
    
# Plots data for varying n
# Input: 

def plot_n(df, ns, ks, epsilons, tests, intercept_eps, iterations, data_function, 
    intercept = False):

    choices =  np.array(np.meshgrid(ks, epsilons, tests, intercept_eps)).T.reshape(-1,4)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, np.shape(choices)[0])))

    for setting in choices:
        filtered = df[(df[:,1] == setting[0]) & (df[:,2] == setting[1]) & 
            (df[:,3] == setting[2]) & (df[:,4] == setting[3])]
        c = next(color)
        plt.plot(filtered[:,0],filtered[:,5],c=c, label = str(setting))
    plt.title("Function: " + data_function.__name__ + ", Iterations: " + str(iterations))
    plt.xlabel("Sample Size")
    plt.ylabel(data_function.__name__)
    plt.legend()
    save_name = "diffn" + str(ns) + str(ks)+ str(epsilons)+ str(tests) + str(intercept_eps)+str(iterations)+".png"
    plt.savefig("./plots/" + save_name)



# Defining a test data generation function
# Inputs: n - sample size

def test_function(n):
    n= int(n)
    slope = 1
    xi = np.linspace(-1,1,num=n)
    yi = slope * xi + np.random.normal(size=n)
    wi = np.random.uniform(0.1, 1, size=n)
    dfi = np.array([xi,yi,wi])
    return dfi


    

if __name__ == "__main__":
    
    t0 = time.time()

    zns = [20,50,80,100,150,200]
    zks = [0,10]
    zes = [1.]
    zts = [0,1,2]
    zies = [0]
    ziter = 500

    df_play = data_gen(zns, zks, zes, zts, zies, ziter, test_function)
    df_play = summary_statistics(df_play, ziter, pstdev)
    plot_n(df_play, zns, zks, zes, zts, zies, ziter, pstdev)
    t1 = time.time()
    print("Time taken: " + str(t1-t0))