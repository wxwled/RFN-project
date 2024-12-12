# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import time
import math
import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
import matplotlib as mpl
from sympy import *

mpl.rcParams.update({
    'text.usetex': True,
    'pgf.preamble': r'\usepackage{bm}',
    "font.family":'serif',
    "font.serif": ['SimSun'],
    "font.size": 15,
    "mathtext.fontset":'cm',
})

random.seed(int(time.time()))

m,N,L,k,r,α = symbols('m N L k r α')
x = symbols('x')
#p_e:p_expose
pe = (1-(1-r/m)**(N*L))**k
attack_point = m*(1-((((N*L)-1)*(3*(N*L)*k-4-(N*L))-(N*L)*(((N*L)-1)*(k-1)*(5*(N*L)*k-7-(N*L)-k))**0.5)/(2*((N*L)*k-1)*((N*L)*k-2)))**(1/(N*L)))

def NL_solve(k,c):
    return math.log((3*k + math.sqrt((k - 1)*(5*k - 1)) - 1)/(2*k**2))/math.log(1-c)

def p_expose_estimate(m,N,L,k,r,expr_num = 10000):
    m_nodes = list(range(1,m+1))
    r_nodes = list(range(1,r+1))
    key_expose_cnt = 0
    for x in range(expr_num):
        delay_expose_count=0
        for l in range(k):
            random_forwarding_path=[]
            for i in range(L):
                for j in range(N):
                    random_forwarding_path.append(random.choices(m_nodes)[0])
            if list(set(r_nodes) & set(random_forwarding_path)):
                delay_expose_count += 1
    
        if delay_expose_count == k:
            key_expose_cnt += 1
    return key_expose_cnt/expr_num

def n_otp1(m0,N0,L0,k0,α0):
    n_attack = round(attack_point.evalf(subs={m:m0,N:N0,L:L0,k:k0})/α0)
    return min(n_attack,m0)

def n_otp2(m,N,L,k,α):
    return round(k/(k*((N*L)-1)/m-math.log(α)))

def p_success_I_estimate(m,N,L,k,α,expr_num = 100000):
    m_nodes = list(range(1,m+1))
    n = n_otp1(m,N,L,k,α)
    key_expose_cnt = 0
    for x in range(expr_num):
        r_nodes = []
        for i in range(1,n+1):
            if random.random()<α:
                r_nodes.append(i)
        delay_expose_count=0
        for l in range(k):
            random_forwarding_path=[]
            for i in range(L):
                for j in range(N):
                    random_forwarding_path.append(random.choices(m_nodes)[0])
            if list(set(r_nodes) & set(random_forwarding_path)):
                delay_expose_count += 1
    
        if delay_expose_count == k:
            key_expose_cnt += 1
    return key_expose_cnt/expr_num

def p_success_II_estimate(m,N,L,k,α,expr_num = 100000):
    m_nodes = list(range(1,m+1))
    n = n_otp2(m,N,L,k,α)
    key_expose_cnt = 0
    
    for x in range(expr_num):
        alarm = False
        r_nodes = []
        for i in range(1,n+1):
            if random.random()<α:
                r_nodes.append(i)
            else:
                alarm = True
        if not alarm:        
            delay_expose_count=0
            for l in range(k):
                random_forwarding_path=[]
                for i in range(L):
                    for j in range(N):
                        random_forwarding_path.append(random.choices(m_nodes)[0])
                if list(set(r_nodes) & set(random_forwarding_path)):
                    delay_expose_count += 1
            if delay_expose_count == k:
                key_expose_cnt += 1
    return key_expose_cnt/expr_num



# m,N,k,r = 100,2,10,10
# p_expose_list_expr = [p_expose_estimate(m,N,k,r,i) for i in range(11)]
# p_expose_list_theo = [pe.evalf(subs={'m':100,'N':10,'k':10,'r':10,'L':i}) for i in range(11)]
# plt.plot(p_expose_list_expr,label = 'experiment')
# plt.plot(p_expose_list_theo,label = 'theory')
# plt.title('{m:100,N:10,k:10,r:10}')
# plt.legend()
# plt.xlabel('L')
# plt.ylabel('p_expose')
# plt.savefig('1',format='svg')
# plt.show()

# m,k,r = 100,10,10
# p_expose_list_expr_with_N=[]
# for N in range(2,7):
#     p_expose_list_expr_with_N.append([p_expose_estimate(m,N,k,r,i) for i in range(11)])
    
# p_expose_list_expr_with_N_plt = np.array(p_expose_list_expr_with_N)  


# plt.hlines([0.0365],0,10,linestyles='dashed',colors='red',label='$NL=13.15$')
# plt.plot(p_expose_list_expr_with_N_plt.T,label=['$N=$'+str(i) for i in range(2,7)])
# plt.legend()
# plt.title('${k:10,c:0.1}$')
# plt.xlabel('$L$')
# plt.ylabel('$p_{expose}$')
# plt.savefig('1.svg',format='svg')
# plt.show()