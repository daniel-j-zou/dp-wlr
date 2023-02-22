# This module includes functions that run complete weighted linear regression algorithms

import numpy as np
import medians
from slope_gen import create_slope_matrix

# Runs DP weighted linear regression
# Inputs: dataset, k - number of matchings, n - sample size of dataset, epsilon - total privacy
# weighted - whether weighted or not, intercept - whether intercept, 
# intercept_eps - proportion of privacy to find intercept
# 
# Output: DP slope

def private_wlr(df, k, n, epsilon, weighted = True, intercept = False, intercept_eps = 0
    , lower_bound = -50, upper_bound = 50):

    # Change all weights to 1 if not weighted

    if not weighted:
        df[1] = np.ones(n)

    # Create slope matrix    

    slope_matrix = create_slope_matrix(df, k, n, lower_bound, upper_bound)

    # Calculate private median
    if k == 0:
        k = n-1
    privacy_factor = 2 * k
    private_val = medians.private_weighted_median_exp(slope_matrix, epsilon/privacy_factor)

    return private_val




# Runs non-private weighted linear regression
# Inputs: dataset, k - number of matchings, n - sample size of dataset,
# weighted - whether weighted or not, intercept - whether intercept, 
# 
# Output: slope

def non_private_wlr(df, k, n, weighted = True, intercept = False, 
    lower_bound = -50, upper_bound = 50):

    # Change all weights to 1 if not weighted

    if not weighted:
        df[1] = np.ones(n)
    
    slope_matrix = create_slope_matrix(df, k, n, lower_bound, upper_bound)
    
    non_private_val = medians.non_private_weighted_median(slope_matrix)
    
    return non_private_val


