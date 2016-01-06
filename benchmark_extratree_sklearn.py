
import time
from sklearn.tree import ExtraTreeRegressor as ETR
from sklearn.ensemble import ExtraTreesRegressor as ETRs, RandomForestRegressor as RFR
from sklearn.datasets import load_boston, make_friedman1
import numpy as np
from numpy import random

def make_friedman1_random_attr(n_samples, n_features):
    n_attrs = 10
    X = random.rand(n_samples,n_attrs)
    Y = np.zeros(shape=(n_samples,))

    sample_index_arr = np.arange(n_samples)
    attr_index_arr = np.arange(n_attrs)
    # random.shuffle(attr_index_arr)
    cur_attr_index_arr = attr_index_arr[0:5]

    for row_index,row in enumerate(X):
        x0 = row[cur_attr_index_arr[0]]
        x1 = row[cur_attr_index_arr[1]]
        x2 = row[cur_attr_index_arr[2]]
        x3 = row[cur_attr_index_arr[3]]
        x4 = row[cur_attr_index_arr[4]]
        Pi = np.pi
        E = np.exp(1)

        Y[row_index] = 10*np.sin(Pi*x0*x1) + 20*(x2-0.5)*(x2-0.5) + 10*x3 + 5*x4 + E

    return X, Y


def make_top_dataset(n_samples, n_attrs):
    X = np.zeros(shape=(n_samples,n_attrs))
    Y = np.zeros(shape=(n_samples,))

    sample_index_arr = np.arange(n_samples)
    random.shuffle(sample_index_arr)

    for i in xrange(n_samples):
        sample_index = sample_index_arr[i]
        row = X[sample_index]
        for j,x in enumerate(row):
            row[j] = i
        Y[sample_index] = i+0.2

    return X, Y

def test():
    X,Y = make_top_dataset(10, 3)
    print X, Y

def score(ext, tX, tY):
    pY = ext.predict(tX)
    return np.sum((tY - pY)**2)/len(pY)

def main():
    # X,Y = make_top_dataset(100000,30)
    X, Y = make_friedman1_random_attr(n_samples=100000, n_features=10)
    tX, tY = make_friedman1_random_attr(n_samples=100, n_features=10)

    start_time = time.time()

    ext = ETRs(max_features=None, n_estimators=100, min_samples_split=1, n_jobs=-1)
    # ext = RFR(max_features=None, n_estimators=100, min_samples_split=1, n_jobs=-1)
    ext.fit(X, Y)

    elapsed_time = time.time() - start_time
    print elapsed_time

    print score(ext, tX, tY)

if __name__ == '__main__':
    main()
    # test()
