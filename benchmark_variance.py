

import numpy as np
from numpy import random

def gen_data():
    # data = [100,20,24,83,30,84,36,2,36,73,10]
    data = random.random(10)
    data += 1e+9
    data *= 1e+9
    print "std var: ", np.var(data)
    # data = minmax_scale(data)
    print data
    return np.asarray(data, dtype=np.float128)

def minmax_scale(X):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    return X_std

def online_variance(data):
    n = 0
    mean = 0.0
    M2 = 0.0
     
    for x in data:
        n += 1
        delta = x - mean
        mean += delta/n
        M2 += delta*(x - mean)

    if n < 2:
        return float('nan');
    else:
        return M2 / (n)

def bessel_variance(data):
    sum_x = np.sum(data)
    sum_x2 = np.sum(data**2)
    n = len(data)

    stdvar = (sum_x2 - sum_x**2/n)/n
    return stdvar

def stream_variance(data):
    sum_x = np.sum(data)
    sum_x2 = np.sum(data**2)
    n = len(data)

    mean = sum_x / n
    stdvar = np.abs(sum_x2/n - mean**2)
    return stdvar

def online_variance_block_sum(data):
    indices = random.choice(np.arange(len(data)), 2)
    cond_arr = data[indices]
    cond_arr = sorted(cond_arr)
    print cond_arr
    block_arr = [[] for _ in cond_arr] + [[]]

    for d in data:
        # if d < cond_arr[0]:
        #     block_arr[0].append(d)
        # elif d>cond_arr[-1]:
        #     block_arr[-1].append(d)
        # else:
        #     block_arr[1].append(d)
        for i,cond in enumerate(cond_arr):
            if d < cond:
                block_arr[i].append(d)
                break
        else:
            block_arr[-1].append(d)

    stdvar = 0
    print block_arr
    for block in block_arr:
        result = online_variance(block)
        if result != float("nan"):
            stdvar += result

    return stdvar

def main():
    data = gen_data()
    print online_variance(data)
    # print online_variance_block_sum(data)
    print stream_variance(data)
    print bessel_variance(data)

if __name__ == '__main__':
    main()


