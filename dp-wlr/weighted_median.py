import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, pstdev

# np.random.seed(0)
n = 30
x = np.linspace(-1,1,num=n)
y = x + np.random.normal(size=n)
w = np.random.uniform(0, 1, size=n)
df = np.array([x,y,w])
lower_bound = -50
upper_bound = 50

# Weighted Theil-Sen

# constructs a sorted slope-weight matrix from a dataframe of [x_values,y_values,weights]
# truncates to [lower_bound,upper_bound] bounds

def create_slope_matrix(df, k):
    n = np.shape(df)[1]
    slope_list = [lower_bound,upper_bound]
    weight_list = [0,0]

    if k == n:
        # appends all slopes and weights, then sorts based on slope

        for i in range(n):
            for j in range(i+1,n):
                if (df[0][j] != df[0][i]):
                    slope_list.append( (df[1][j] - df[1][i]) / (df[0][j] - df[0][i]))
                    weight_list.append(df[2][i] * df[2][j])
        

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
    n = np.shape(df)[1]
    return(compute_weighted_median(create_slope_matrix(df, n)))

#print(compute_weighted_median(df_slope))


# Takes in a slope-weight matrix, computes utility for each point, uses the Gumbel method
# Outputs a single DP value

def exponential_preprocessed(df, epsilon):
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

def dp_wlr_preprocessed(df, epsilon  = 1, k = 10):
    slope_matrix = create_slope_matrix(df, k)
    private_val = exponential_preprocessed(slope_matrix, epsilon/2/k)
    return private_val

""" 
def create_slope_matrix_k(df,k):
    n = np.shape(df)[1]
    x = df[0]
    y = df[1]
    weights = df[2]
    z = np.arange(n)
    z = np.random.permutation(z)
    
    a = np.random.choice(n-1, k, replace=False)
    # compute n/2 estimates
    
    for j in a:
        p = z[j]
        q = z[n-1]
        x_delta = float(x[q]-x[p])
        if x_delta != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
            slope = float(y[q]-y[p])/ float(x_delta) # compute slope between two points
            for m in range(len(xnew)):
                xnew_ests[m].append(slope*np.array(xnew[m])+(ymean-slope*xmean))
                
        for i in range(1,int((n-1)/2+1)):
            p = z[(j-i)%(n-1)]
            q = z[(j+i)%(n-1)]
            x_delta = float(x[q]-x[p])
            if x_delta != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
                slope = float(y[q]-y[p])/ float(x_delta) # compute slope between two points
                xmean = (x[q]+x[p])/2
                ymean = (y[q]+y[p])/2
                for m in range(len(xnew)):
                    xnew_ests[m].append(slope*np.array(xnew[m])+(ymean-slope*xmean))
 """

# Test with full

def monte_carlo_test(n, slope, tries, epsilon, k):
    output_list = []
    public_list = []
    non_weight_list = []
    for i in range(tries):
        xi = np.linspace(-1,1,num=n)
        yi = slope * xi + np.random.normal(size=n)
        wi = np.random.uniform(0.1, 1, size=n)
        wi_non = [1] * n
        dfi = np.array([xi,yi,wi])
        dfi_non = np.array([xi,yi,wi_non])
        output_list.append(dp_wlr_preprocessed(dfi, epsilon, n))
        public_list.append(public_wlr(dfi))
        non_weight_list.append(dp_wlr_preprocessed(dfi_non, epsilon, n))
    print("private_mean", mean(output_list))
    print("private_sd", pstdev(output_list))
    print("non_private_mean", mean(public_list))
    print("non_private_sd", pstdev(public_list))
    print("non_weighted_mean", mean(non_weight_list))
    print("non_weighted_sd", pstdev(non_weight_list))


# Generates Plot with different ns
def monte_carlo_gen(ns, slope, tries, epsilon):
    private_means = list()
    private_sds = list()
    non_private_means = list()
    non_private_sds = list()
    non_weighted_means = list()
    non_weighted_sds = list()
    for n in ns:
        output_list = []
        public_list = []
        non_weight_list = []
        for i in range(tries):
            xi = np.linspace(-1,1,num=n)
            yi = slope * xi + np.random.normal(size=n)
            wi = np.random.uniform(0.1, 1, size=n)
            wi_non = [1] * n
            dfi = np.array([xi,yi,wi])
            dfi_non = np.array([xi,yi,wi_non])
            output_list.append(dp_wlr_preprocessed(dfi, epsilon, n))
            public_list.append(public_wlr(dfi))
            non_weight_list.append(dp_wlr_preprocessed(dfi_non, epsilon, n))
        private_means.append(mean(output_list))
        private_sds.append(pstdev(output_list))
        non_private_means.append(mean(public_list))
        non_private_sds.append(pstdev(public_list))
        non_weighted_means.append(mean(non_weight_list))
        non_weighted_sds.append(pstdev(non_weight_list))
    output = np.array([ns, private_means, private_sds, non_private_means, non_private_sds, non_weighted_means, non_weighted_sds])
    rounded_output = np.around(output, 3)
    np.save('data' + str([ns[0],ns[-1]]) + str(tries) + "_" + str(epsilon), rounded_output)
    print(rounded_output)

# plot for different n's    

def monte_carlo_plot(min,max,tries,eps):
    data = np.load('data[' + str(min) + ', ' + str(max) + ']' + str(tries) + '_' + str(eps) + '.npy')
    plt.plot(data[0],data[1],'r--', label = "priv_weighted")
    plt.plot(data[0], data[3], 'g--', label = "non_priv_weighted")
    plt.plot(data[0], data[5], 'b--', label = "priv_non_weighted")
    plt.title("Mean: Epsilon = " + str(eps) + ', Reps = ' + str(tries))
    plt.xlabel("Sample Size")
    plt.ylabel("Mean of Output")
    plt.legend()
    plt.savefig('mean[' + str(min) + ',' + str(max) + ']' + str(tries) + '_' + str(eps) + '.png', )
    plt.clf()
    plt.plot(data[0],data[2],'r--', label = "priv_weighted")
    plt.plot(data[0], data[4], 'g--', label = "non_priv_weighted")
    plt.plot(data[0], data[6], 'b--', label = "priv_non_weighted")
    plt.title("Standard Deviation: Epsilon = " + str(eps) + ', Reps = ' + str(tries))
    plt.xlabel("Sample Size")
    plt.ylabel("Standard Deviation of Output")
    plt.legend()
    plt.savefig('sd[' + str(min) + ',' + str(max) + ']' + str(tries) + '_' + str(eps) + '.png', )



# generates plot with different ks
def monte_carlo_gen_k(n, slope, tries, epsilon,ks):
    private_means = list()
    private_sds = list()
    non_private_means = list()
    non_private_sds = list()
    non_weighted_means = list()
    non_weighted_sds = list()
    for k in ks:
        output_list = []
        public_list = []
        non_weight_list = []
        for i in range(tries):
            xi = np.linspace(-1,1,num=n)
            yi = slope * xi + np.random.normal(size=n)
            wi = np.random.uniform(0.1, 1, size=n)
            wi_non = [1] * n
            dfi = np.array([xi,yi,wi])
            dfi_non = np.array([xi,yi,wi_non])
            output_list.append(dp_wlr_preprocessed(dfi, epsilon, k))
            public_list.append(public_wlr(dfi))
            non_weight_list.append(dp_wlr_preprocessed(dfi_non, epsilon, k))
        private_means.append(mean(output_list))
        private_sds.append(pstdev(output_list))
        non_private_means.append(mean(public_list))
        non_private_sds.append(pstdev(public_list))
        non_weighted_means.append(mean(non_weight_list))
        non_weighted_sds.append(pstdev(non_weight_list))
    output = np.array([ks, private_means, private_sds, non_private_means, non_private_sds, non_weighted_means, non_weighted_sds])
    rounded_output = np.around(output, 3)
    np.save('datak' + str([ks[0],ks[-1]]) + str(tries) + "_" + str(epsilon), rounded_output)
    print(rounded_output)    

def monte_carlo_plot_k(min,max,tries,eps):
    data = np.load('datak[' + str(min) + ', ' + str(max) + ']' + str(tries) + '_' + str(eps) + '.npy')
    plt.plot(data[0],data[1],'r--', label = "priv_weighted")
    plt.plot(data[0], data[3], 'g--', label = "non_priv_weighted")
    plt.plot(data[0], data[5], 'b--', label = "priv_non_weighted")
    plt.title("Mean: Epsilon = " + str(eps) + ', Reps = ' + str(tries))
    plt.xlabel("k")
    plt.ylabel("Mean of Output")
    plt.legend()
    plt.savefig('meank[' + str(min) + ',' + str(max) + ']' + str(tries) + '_' + str(eps) + '.png', )
    plt.clf()
    plt.plot(data[0],data[2],'r--', label = "priv_weighted")
    plt.plot(data[0], data[4], 'g--', label = "non_priv_weighted")
    plt.plot(data[0], data[6], 'b--', label = "priv_non_weighted")
    plt.title("Standard Deviation: Epsilon = " + str(eps) + ', Reps = ' + str(tries))
    plt.xlabel("k")
    plt.ylabel("Standard Deviation of Output")
    plt.legend()
    plt.savefig('sdk[' + str(min) + ',' + str(max) + ']' + str(tries) + '_' + str(eps) + '.png', )




if __name__ == "__main__":
    #monte_carlo_gen_k(100,0,100, 1,range(1,101))
    #monte_carlo_plot_k(1,100,100,1)
    print(create_slope_matrix(df,30))
