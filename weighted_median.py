import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, pstdev

# Mini-example

# np.random.seed(0)
n = 30
x = np.linspace(-1,1,num=n)
y = x + np.random.normal(size=n)
w = np.random.uniform(0, 1, size=n)
df = np.array([x,y,w])
lower_bound = -30
upper_bound = 30

# Weighted Theil-Sen

# constructs a sorted slope-weight matrix from a dataframe of [x_values,y_values,weights]
# truncates to [lower_bound,upper_bound] bounds

def create_slope_matrix(df):
    n = np.shape(df)[1]
    slope_list = [lower_bound,upper_bound]
    weight_list = [0,0]

    # appends all slopes and weights, then sorts based on slope

    for i in range(n):
        for j in range(i+1,n):
            slope_list.append( (df[1][j] - df[1][i]) / (df[0][j] - df[0][i]))
            weight_list.append(df[2][i] * df[2][j])
    out_df = np.array([slope_list, weight_list])
    out_df = out_df[:, out_df[0, :].argsort()]
    
    # truncates to bounds

    if out_df[0][0] < lower_bound:
        lower_bound_index = np.searchsorted(out_df[0], lower_bound)
        out_df[0][:lower_bound_index] = [lower_bound] * lower_bound_index

    if out_df[0][-1] > upper_bound:
        upper_bound_index = np.searchsorted(out_df[0], upper_bound)
        out_df[0][upper_bound_index:] = [upper_bound] * int(n * (n-1) / 2 - upper_bound_index+ 2)
    
    return out_df

df_slope = create_slope_matrix(df)    


# inputs a slope-weight matrix and outputs the non-private weighted median.    

def compute_weighted_median(df):
    n = np.shape(df)[1]
    cumsum = np.cumsum(df[1])
    totalsum = cumsum[n-1]
    index_first_match = next(
        (index for index, item in enumerate(cumsum) if item > totalsum/2)
        )
    return df[0][index_first_match]

# Non-private Weighted Theil-Sen

def public_wlr(df):
    return(compute_weighted_median(create_slope_matrix(df)))

#print(compute_weighted_median(df_slope))


# Takes in a slope-weight matrix, computes utility for each point, uses the Gumbel method
# Outputs a single DP value

def exponential_preprocessed(df, epsilon = 1):
    n = np.shape(df)[1]
    slopes = df[0]
    max_noisy_score = float('-inf')
    arg_max_noisy_score = -1
    cumsum = np.cumsum(df[1])
    totalsum = cumsum[-1]

    for i in range(1,n):
        interval_length = slopes[i] - slopes[i-1]
        if interval_length == 0:
            None
        else: 
            log_interval_length = math.log(interval_length)
            utility = abs(cumsum[i]- totalsum/2)
            score = log_interval_length - epsilon * utility
            noisy_score = score + np.random.gumbel(0,1)
            if noisy_score > max_noisy_score:
                max_noisy_score = noisy_score
                arg_max_noisy_score = i
    # print(arg_max_noisy_score, max_noisy_score)

    left_bound = slopes[arg_max_noisy_score - 1]
    right_bound = slopes[arg_max_noisy_score]
    private_val = np.random.uniform(left_bound, right_bound)

    return private_val


# Complete Private WLR with weighted Theil-Sen

def dp_wlr_preprocessed(df, epsilon  = 1):
    slope_matrix = create_slope_matrix(df)
    private_val = exponential_preprocessed(slope_matrix, epsilon)
    return private_val

# print(dp_wlr_preprocessed(df))


def monte_carlo_test(n, slope, tries, epsilon):
    output_list = []
    public_list = []
    non_weight_list = []
    for i in range(tries):
        xi = np.linspace(-1,1,num=n)
        yi = slope * xi + np.random.normal(size=n)
        wi = np.random.uniform(0.5, 1, size=n)
        wi_non = [1] * n
        dfi = np.array([xi,yi,wi])
        dfi_non = np.array([xi,yi,wi_non])
        output_list.append(dp_wlr_preprocessed(dfi, epsilon/(n-1)))
        public_list.append(public_wlr(dfi))
        non_weight_list.append(dp_wlr_preprocessed(dfi_non, epsilon/(n-1)))
    print("private_mean", mean(output_list))
    print("private_sd", pstdev(output_list))
    print("non_private_mean", mean(public_list))
    print("non_private_sd", pstdev(public_list))
    print("non_weighted_mean", mean(non_weight_list))
    print("non_weighted_sd", pstdev(non_weight_list))

monte_carlo_test(30,0,1000, 1)
