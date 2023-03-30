# This module includes functions that run complete weighted linear regression algorithms

import numpy as np
import medians
from slope_gen import create_slope_matrix, create_point_matrix
import statsmodels.api as sm
from statistics import mean, pstdev, median

# Runs DP weighted linear regression
# Inputs: dataset: x,y,weights
# , k - number of matchings, n - sample size of dataset, epsilon - total privacy
# weighted - whether weighted or not, intercept - whether intercept, 
# intercept_eps - proportion of privacy to find intercept
# 
# Output: DP slope

def private_wlr(df, k, n, epsilon, weighted = True, intercept = False, intercept_eps = 0
    , lower_bound = -50, upper_bound = 50):

    # Change all weights to 1 if not weighted

    if not weighted:
        df[2] = np.ones(n)

    # Create slope matrix    

    slope_matrix = create_slope_matrix(df, k, n, lower_bound, upper_bound)

    if not intercept:

        # Calculate private median
        if k == 0:
            k = n-1
        privacy_factor = 2 * k
        private_val = medians.private_weighted_median_exp(slope_matrix, epsilon/privacy_factor)
        new_x = sm.add_constant(df[0])
        model = sm.OLS(df[1], new_x)
        results = model.fit()
        return [private_val, 0, (results.params)[1], (results.params)[0]]

    if intercept:

        slope_epsilon = (100-intercept_eps) * epsilon / 100
        intercept_epsilon = intercept_eps * epsilon / 100

        # Calculate private median of slope
        if k == 0:
            k = n-1
        privacy_factor = 2 * k
        private_val = medians.private_weighted_median_exp(slope_matrix, slope_epsilon/privacy_factor)

        # Calculate private intercept
        residuals = df[1] - private_val * df[0]
        residuals = np.append(residuals, [-0.5,1.5])
        res_weights = df[2]
        res_weights = np.append(res_weights, [0,0])

        # sort according to residuals

        out_df = np.array([residuals, res_weights])
        out_df = out_df[:, out_df[0, :].argsort()]    

        # truncates to bounds

        out_df[0] = np.clip(out_df[0], -0.5,1.5)
        
        
        # Assume -0.5 to 1.5, so range is 2
        private_intercept = medians.private_weighted_median_exp(out_df, intercept_epsilon / 2)
        new_x = sm.add_constant(df[0])
        model = sm.OLS(df[1], new_x)
        results = model.fit()
        return [private_val, private_intercept, (results.params)[1], (results.params)[0]]


    




# Runs non-private weighted linear regression
# Inputs: dataset, k - number of matchings, n - sample size of dataset,
# weighted - whether weighted or not, intercept - whether intercept, 
# 
# Output: slope

def non_private_wlr(df, k, n, weighted = True, intercept = False, 
    lower_bound = -50, upper_bound = 50):

    # Change all weights to 1 if not weighted

    if not weighted:
        df[2] = np.ones(n)
    
    slope_matrix = create_slope_matrix(df, k, n, lower_bound, upper_bound)

    if not intercept:
    
        non_private_val = medians.non_private_weighted_median(slope_matrix)
        new_x = sm.add_constant(df[0])
        model = sm.OLS(df[1], new_x)
        results = model.fit()
        return [non_private_val, 0, (results.params)[1], (results.params)[0]]


    if intercept:

        non_private_val = medians.non_private_weighted_median(slope_matrix)

        # Calculate private intercept
        
        residuals = df[1] - non_private_val * df[0]
        res_weights = df[2]
        # sort according to slopes

        out_df = np.array([residuals, res_weights])
        out_df = out_df[:, out_df[0, :].argsort()]    

        # truncates to bounds
        out_df[0] = np.clip(out_df[0], -0.5,1.5)
        # Assume -0.5 to 1.5, so range is 2
        non_private_intercept = medians.non_private_weighted_median(out_df)
        # non_private_intercept = median(residuals)
        new_x = sm.add_constant(df[0])
        model = sm.OLS(df[1], new_x)
        results = model.fit()
        return [non_private_val,non_private_intercept,(results.params)[1], (results.params)[0]]




# Runs private_point_wlr as in the 2020 paper
# Inputs: dataset, k - number of matchings, n - sample size of dataset,
# weighted - whether weighted or not, intercept - whether intercept, 
# 
# Output: private est, 0, wlr est, 0

def private_point_wlr(df, k, n, epsilon, weighted = True, intercept = False, 
                      lower_bound = 0, upper_bound = 1):

    # Change all weights to 1 if not weighted

    if not weighted:
        df[2] = np.ones(n)
    
    point_matrix = create_point_matrix(df, k, n, lower_bound, upper_bound)

    if k == 0:
            k = n-1
    privacy_factor = 2 * k

    private_val = medians.private_weighted_median_exp(point_matrix, epsilon/(privacy_factor*2))
    new_x = sm.add_constant(df[0])
    model = sm.OLS(df[1], new_x)
    results = model.fit()
    return [private_val, 0, (results.params)[1] * 0.25 + (results.params)[0], 0]

    
