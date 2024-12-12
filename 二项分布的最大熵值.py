# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:26:35 2020

@author: wxw
"""
import numpy as np
from scipy.special import comb

def entropy(f):
    s = 0
    for i in f:
        if i != 0:
            s += -i*np.log2(i)
    return s

N=2
p=0.5
f = np.array([comb(N,i)*p**(i)*(1-p)**(N-i) for i in range(N+1)])
print(f)
print(entropy(f))

#p=0.5 熵最大