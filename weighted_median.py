import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Mini-example

np.random.seed(0)
n = 5
x = np.linspace(-1,1,num=n)
y = x + np.random.normal(size=n)
w = np.random.uniform(0, 1, size=n)
df = np.array([x,y,w])

# Weighted Theil-Sen

# constructs a sorted slope-weight matrix

def create_slope_matrix(df):
    n = np.shape(df)[1]
    slope_list = []
    weight_list = []
    for i in range(n):
        for j in range(i+1,n):
            slope_list.append( (df[1][j] - df[1][i]) / (df[0][j] - df[0][i]))
            weight_list.append(df[2][i] * df[2][j])
    out_df = np.array([slope_list, weight_list])
    out_df = out_df[:, out_df[0, :].argsort()]
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

def pub_wts(df):
    return(compute_weighted_median(create_slope_matrix(df)))

#print(compute_weighted_median(df_slope))

# Calculates the utility based on -|#above-#below|, takes in a slope-weight matrix

def util_edit(slopes, input, cumsum, totalsum,n ):
    below_index = int(list(filter(lambda i: i >= input, slopes))[0] - 2)
    above_list = list(filter(lambda i: i > input, slopes))
    above_list.append(n+1)
    above_index = int(above_list[0] - 2)
    return(totalsum - cumsum[above_index] - cumsum[below_index])

def util_naive(slopes, input, cumsum, totalsum, n):
    below_index = int(list(filter(lambda i: i >= input, slopes))[0] - 2)
    above_list = list(filter(lambda i: i > input, slopes))
    above_list.append(n+1)
    above_index = int(above_list[0] - 2)
    lower_sum = cumsum[below_index]
    upper_sum = cumsum[above_index]
    if lower_sum < totalsum / 2 and upper_sum > totalsum/2: return 0
    distance = min(abs(lower_sum - totalsum / 2), abs(upper_sum - totalsum/2))
    return(-distance)




def exponential_utility(df):
    n = np.shape(df)[1]
    cumsum = np.cumsum(df[1])
    totalsum = cumsum[n-1]
    min = df[0][0]
    max = df[0][n-1]
    slopes = np.linspace(min,max,1000)
    utils = [util_naive(slopes, slope, cumsum, totalsum,n) for slope in slopes]
    return(np.array([slopes, utils]))

df_util = exponential_utility(df_slope)

def dp_wls(df, epsilon = 1):
    n = np.shape(df)[1]
    slopes = df[0][0:n-2]
    utils = df[1][0:n-2]
    probs = [math.exp(epsilon * util/2) for util in utils]
    sum_probs = sum(probs)
    norm_probs = [item/sum_probs for item in probs]
    return(np.random.choice(slopes,10,p=norm_probs))
    
print(dp_wls(df_util))
