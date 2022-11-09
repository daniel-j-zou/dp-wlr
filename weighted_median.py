import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Mini-example

np.random.seed(0)
n = 100
x = np.linspace(-5,5,num=n)
y = x + np.random.normal(size=n)
w = np.random.uniform(0, 1, size=n)
df = np.array([x,y,w])
print(df)

# Weighted Theil-Sen

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

def compute_weighted_median(df):
    n = np.shape(df)[1]
    cumsum = np.cumsum(df[1])
    totalsum = cumsum[n-1]
    index_first_match = next(
        (index for index, item in enumerate(cumsum) if item > totalsum/2)
        )
    return df[0][index_first_match]

print(compute_weighted_median(create_slope_matrix(df)))

