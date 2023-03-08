# This module contains functions that generate slope-weight matrices according to the requirements

import math
import numpy as np


# Creates a sorted Slope-Weight Matrix for a specific k, with appended lower and upper bounds
# If k == 0, do complete graph
# If k < n, do k samplings with inclusion
# Inputs: dataset, k - number of samplings, n - sample size of dataset

def create_slope_matrix(df, k, n, lower_bound, upper_bound):

    # Initializes with lower and upper bounds

    slope_list = [lower_bound,upper_bound]
    weight_list = [0,0]

    # Complete Graph Case

    if k == 0:

        # appends all slopes and weights, then sorts based on slope

        for i in range(n):
            for j in range(i+1,n):
                if (df[0][j] != df[0][i]):
                    slope_list.append( (df[1][j] - df[1][i]) / (df[0][j] - df[0][i]))
                    weight_list.append(df[2][i] * df[2][j])

    # k-Match Case    

    else: 
        for i in range(k):
            ordering = np.random.choice(range(n),n,replace=False, p=(df[2] / df[2].sum()))
            for j in range(n // 2):
                i_1 = ordering[2 * j]
                i_2 = ordering[2 * j + 1]
                if (df[0][i_1] != df[0][i_2]):
                    slope_list.append( (df[1][i_1] - df[1][i_2]) / (df[0][i_1] - df[0][i_2]))
                    weight_list.append(df[2][i_1] * df[2][i_2])

    # sort according to slopes

    out_df = np.array([slope_list, weight_list])
    out_df = out_df[:, out_df[0, :].argsort()]    

    # truncates to bounds

    out_df[0] = np.clip(out_df[0], lower_bound, upper_bound)
    
    return out_df

