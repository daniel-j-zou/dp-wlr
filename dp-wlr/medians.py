# This module provides functions for all median-related algorithms

import math
import numpy as np

# Computes a public weighted median
# Inputs: slope-weight matrix
# Output: median

def non_private_weighted_median(df):
    # n represents size of slope matrix
    n = np.shape(df)[1]
    cumsum = np.cumsum(df[1])
    totalsum = cumsum[n-1]
    index_first_match = next(
        (index for index, item in enumerate(cumsum) if item > totalsum/2)
        )
    return df[0][index_first_match]

# Computes a differentially private weighted median with the exponential mechanism
# Input: slope-weight matrix, privacy_factor
# Outputs a single DP value
# Uses Gumbel Distribution method

def private_weighted_median_exp(df, privacy_factor):

    n = np.shape(df)[1]
    slopes = df[0]
    max_noisy_score = float('-inf') # indicates the maximum noisy score
    arg_max_noisy_score = -1        # indicates the interval with the maximum noisy score

    cumsum = np.cumsum(df[1])
    totalsum = cumsum[-1]

    for i in range(1,n):
        interval_length = slopes[i] - slopes[i-1]
        if interval_length == 0:
            None
        else: 
            log_interval_length = math.log(interval_length)
            utility = abs(cumsum[i]- totalsum/2)
            score = log_interval_length - privacy_factor * utility
            noisy_score = score + np.random.gumbel(0,1)
            if noisy_score > max_noisy_score:
                max_noisy_score = noisy_score
                arg_max_noisy_score = i
    # print(arg_max_noisy_score, max_noisy_score)

    left_bound = slopes[arg_max_noisy_score - 1]
    right_bound = slopes[arg_max_noisy_score]
    private_val = np.random.uniform(left_bound, right_bound)

    return private_val


