# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:44:10 2021

@author: wxw
"""
import math

def f(N,i):
    if i>N-i:
        i=N-i
    x = sum([math.log(i/j+1) for j in range(1,N-i+1)])-N*math.log(2)
    return math.exp(x)

def H(N):
    return sum([-f(N,i)*math.log2(f(N,i)) for i in range(N+1)])

import matplotlib.pyplot as plt
plt.plot([H(i) for i in range(500)])
plt.xlabel('N')
plt.ylabel('H/bit')
plt.show()