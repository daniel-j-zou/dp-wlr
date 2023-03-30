# This module includes all functions related to Monte-Carlo, both in generating data, 
# summarizing data, and plotting it

import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, pstdev
from run_wlr import private_wlr, non_private_wlr, private_point_wlr
import os
import statsmodels.api as sm

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

    df = np.array(np.meshgrid(ns, ks, epsilons, tests, intercept_eps, [0], [0], [0], [0])).T.reshape(-1,9)
    df = np.repeat(df, repeats = iterations, axis = 0)

    for i in range(np.shape(df)[0]):
        params = df[i]
        params = [int(x) for x in params]
        dfi = data_function(params[0])

        # private and weighted

        if params[3] == 0:
            inter = private_wlr(dfi, params[1], params[0], params[2], True, 
                intercept, params[4], lower_bound, upper_bound)
            df[i][5] = inter[0]
            df[i][6] = inter[1]
            df[i][7] = inter[2]
            df[i][8] = inter[3]
        
        # private and non-weighted

        if params[3] == 1: 
            inter = private_wlr(dfi, params[1], params[0], params[2], False, 
                intercept, params[4], lower_bound, upper_bound)
            df[i][5] = inter[0]
            df[i][6] = inter[1]
            df[i][7] = inter[2]
            df[i][8] = inter[3]

        # non-private and weighted

        if params[3] == 2:
            inter = non_private_wlr(dfi, params[1], params[0], True, intercept,
            lower_bound, upper_bound)
            df[i][5] = inter[0]
            df[i][6] = inter[1]
            df[i][7] = inter[2]
            df[i][8] = inter[3]

        # non-private and non-weighted (Theil-Sen)

        if params[3] == 3:
            inter = non_private_wlr(dfi, params[1], params[0], False, intercept,
            lower_bound, upper_bound)
            df[i][5] = inter[0]
            df[i][6] = inter[1]
            df[i][7] = inter[2]
            df[i][8] = inter[3]

        # 2020 paper private point estimate

        if params[3] == 4:
            inter = private_point_wlr(dfi, params[1], params[0], params[2], True, 
                False, 0, 1)
            df[i][5] = inter[0]
            df[i][6] = inter[1]
            df[i][7] = inter[2]
            df[i][8] = inter[3]

        


    df = np.around(df, 3)
    name = 'data' + str(ns) + str(ks) + str(epsilons)+ str(tests) +str(intercept_eps)+ str(iterations)
    np.save(os.path.join('data', name), df)
    np.savetxt('test.csv',df, delimiter=',', fmt='%s')
    return(df)


# Computes summary statistics from datagen
# Inputs: df - data frame or name of data frame, iterations, summary function, intercepts or no 
# 
# Outputs: new array with rows representing different configurations
# columns are: n, k, eps, test_type, inter_eps, slope_summary, inter_summary (0 if none)

def summary_statistics(df, iterations, fn, intercept = False):
    if isinstance(df, str):
        df = np.load(df)

    if fn == "paper":
        n = np.shape(df)[0]
        diff = int(n / iterations)
        newdf = df[[i for i in range(0, n, iterations)],:]
        df = np.transpose(df)
        for i in range(diff):
            slope_data = df[5][i*iterations : (i+1) * iterations - 1]
            slope_summary = (np.quantile(slope_data,0.84) - np.quantile(slope_data,0.16))
            ols_data = df[7][i*iterations : (i+1) * iterations - 1]
            ols_summary = (np.quantile(ols_data,0.84) - np.quantile(ols_data,0.16))
            newdf[i][5] = slope_summary / ols_summary
        if intercept:
            for i in range(diff):
                inter_data = df[6][i*iterations : (i+1) * iterations - 1]
                inter_summary = (np.quantile(inter_data,0.84) - np.quantile(inter_data,0.16))
                newdf[i][6] = inter_summary
        
        newdf = np.around(newdf, 3)
        return(newdf)
    
    if fn == "paper_point":
        n = np.shape(df)[0]
        diff = int(n / iterations)
        newdf = df[[i for i in range(0, n, iterations)],:]
        df = np.transpose(df)
        for i in range(diff):
            if int(df[3][i*iterations]) == 4:
                point_data = df[5][i*iterations : (i+1) * iterations - 1]
                ols_point_data = df[7][i*iterations : (i+1) * iterations - 1]
                point_summary = (np.quantile(point_data,0.84) - np.quantile(point_data,0.16))
                ols_summary = (np.quantile(ols_point_data,0.84) - np.quantile(ols_point_data,0.16))
                newdf[i][5] = point_summary / ols_summary
            else:    
                slope_data = df[5][i*iterations : (i+1) * iterations - 1]
                inter_data = df[6][i*iterations : (i+1) * iterations - 1]
                ols_slope_data = df[7][i*iterations : (i+1) * iterations - 1]
                ols_inter_data = df[8][i*iterations : (i+1) * iterations - 1]
                point_est = slope_data * 0.25 + inter_data
                ols_point_est = ols_slope_data * 0.25 + ols_inter_data
                point_summary = (np.quantile(point_est,0.84) - np.quantile(point_est,0.16))
                ols_summary = (np.quantile(ols_point_est,0.84) - np.quantile(ols_point_est,0.16))
                newdf[i][5] = point_summary / ols_summary
        
        newdf = np.around(newdf, 3)
        np.savetxt('test.csv',newdf, delimiter=',', fmt='%s')
        return(newdf)

    n = np.shape(df)[0]
    diff = int(n / iterations)
    newdf = df[[i for i in range(0, n, iterations)],:]
    df = np.transpose(df)
    for i in range(diff):
        slope_data = df[5][i*iterations : (i+1) * iterations - 1]
        slope_summary = fn(slope_data)
        # ols_slope_data = df[7][i*iterations : (i+1) * iterations - 1]
        # ols_slope_summary = fn(ols_slope_data)
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

def plot_n(df, ns, ks, epsilons, tests, intercept_eps, iterations, data_function_name, 
    intercept = False):

    if isinstance(df, str):
        df = np.load(df)

    choices =  np.array(np.meshgrid(ks, epsilons, tests, intercept_eps)).T.reshape(-1,4)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, np.shape(choices)[0]+2)))
    plt.xscale('log')
    #plt.ylim(0.001,5)
    

    for setting in choices:
        filtered = df[(df[:,1] == setting[0]) & (df[:,2] == setting[1]) & 
            (df[:,3] == setting[2]) & (df[:,4] == setting[3])]
        c = next(color)
        plt.plot(filtered[:,0],filtered[:,5],c=c, label = str(setting))
    
    c = next(color)
    plt.axhline(y = 1, color = 'r', linestyle = '--',linewidth=0.5)
    c = next(color)
    plt.axhline(y = 2, color = 'b', linestyle = '--',linewidth=0.5)
    
    
    plt.title("Function: " + data_function_name + ", Iterations: " + str(iterations))
    plt.xlabel("Sample Size")
    plt.ylabel(data_function_name)
    plt.legend()
    save_name = "diffn" + str(ns) + str(ks)+ str(epsilons)+ str(tests) + str(intercept_eps)+str(iterations)+data_function_name + ".png"
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

# Defining paper data generation function
# x values are by definition in [0,1]
# y values are clipped to [0,1]
# Inputs: n - sample size

def paper_function(n):
    n = int(n)
    slope = 0.5
    intercept = 0.2
    xi = np.random.uniform(0,1,n)
    errors = np.random.normal(0, 0.187, size=n)
    yi = slope * xi + intercept + errors
    yi = np.clip(yi, 0, 1)
    wi = np.random.uniform(0.1, 1, size=n)
    dfi = np.array([xi,yi,wi])
    return dfi


def lenny_dgp(n):
    xi = np.random.normal(size=n)
    
    # Treatment, D
    prob = np.exp(0.5 * xi) / (1 + np.exp(0.5 * xi))
    di = 1 * (np.random.uniform(size=n) < prob)
    
    # Outcome, Y
    yi = xi + np.random.normal(size=n)  # True ATT=0

    wi = np.zeros(n)
    
    wi[di == 1] = 1 / sum(di == 1)
    wi[di == 0] = prob[di == 0] / (1 - prob[di == 0])
    wi[di == 0] = wi[di == 0] / sum(wi[di == 0])

    dfi = np.array([di,yi,wi])
    return dfi
    

if __name__ == "__main__":
    
    t0 = time.time()

    zns = [50,100,300,1000]
    zks = [10]
    zes = [2.]
    zts = [0,1,2,3,4]
    zies = [50] 
    ziter = 100
    intercept_bool = True
    data_name = "D:\Reed College\Thesis\dp-wlr-1\data\data[50, 100, 300, 1000][10][2.0][0, 1, 2, 3][0.5]500.npy"

    df_play = data_gen(zns, zks, zes, zts, zies, ziter, paper_function, intercept_bool, -1,1)
    df_play = summary_statistics(df_play, ziter, "paper_point", intercept_bool)
    plot_n(df_play, zns, zks, zes, zts, zies, ziter, "paper_confint", intercept_bool)
    t1 = time.time()
    print("Time taken: " + str(t1-t0)) 