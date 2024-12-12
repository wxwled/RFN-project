# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:33:44 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

def entropy(f):
    s = 0
    for i in f:
        if i != 0:
            s += -i*np.log2(i)
    return s

def relative_entropy(f,interval):
    return entropy(f) + np.log2(interval)

# sample = np.random.randn(10000)
# n_hist = 1000
# f=np.histogram(sample, n_hist)[0]
# f = f/sum(f)
# print('离散熵：',entropy(f))
# print('连续熵：',relative_entropy(f,n_hist))

def experiment(n_hist, N_sample=10000):
    sample = np.random.randn(N_sample)
    f=np.histogram(sample, n_hist)
    interval = f[1][1]-f[1][0]
    # print(interval)
    f = f[0]/sum(f[0])
    # print(f)
    return entropy(f), relative_entropy(f,interval)
    # print('离散熵：',entropy(f))
    # print('连续熵：',relative_entropy(f,n_hist))

n_hists = np.arange(1,1000,10)
y = [experiment(i) for i in n_hists]
plt.plot(n_hists, y)
plt.show()


