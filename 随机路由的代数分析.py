# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:34:50 2020

@author: wxw
"""

import graph
import numpy as np
import matplotlib.pyplot as plt
import sympy

#矩阵快速幂
def fast_power(P,power):
    if power == 0:
        return 1
    if power % 2:
        return P*fast_power(P,power-1)
    else:
        return fast_power(P,power//2)**2

#给定节点网络和转发策略，给出概率转移向量和矩阵
def get_sPt(G,p):
    x = sympy.Symbol('x')
    s=[]
    for i in range(1,G.num-1):
        s.append(p[0,i]*x**G.adj[0,i])
    s = sympy.Matrix(s).T
    P=[]
    for i in range(1,G.num-1):
        pi=[]
        for j in range(1,G.num-1):
            pi.append(p[i,j]*x**G.adj[i,j])    
        P.append(pi)
    P = sympy.Matrix(P)
    t=[]
    for i in range(1,G.num-1):
        t.append(x**G.adj[i,G.num-1])
    t = sympy.Matrix(t)
    return s,P,t

#获取时延多项式s中的时延分布
def coefficient_get(s):
    s = s.replace(' ','')
    if 'x' in s:
        s = s.split('*')
        return float(s[0]),float(s[-1])
    else:
        return float(s),0

#计算分布f的信息熵    
def entropy(f):
    s = 0
    for i in f:
        if i != 0:
            s += -i*np.log2(i)
    return s

#给定节点网络、转发策略和转发次数，给出测量时延的分布
def distribution_calculate(G,p,steps):
    s,P,t = get_sPt(G,p)
    #计算量最大的位置,用矩阵快速幂来提高算法效率
    px = sympy.expand(s*fast_power(P,steps-1)*t)[0]
#    print('分布多项式:',px)
    px = str(px).split('+')
    di=[]
    pi=[]
    for s in px[::-1]:
        p,d = coefficient_get(s)
        pi.append(p)
        di.append(str(d))
    return di,pi


#z_num表示中间节点数，z_steps表示最大转发次数
#G为包含Alice和Bob在内的节点网络情况
def distribution_calculate_show(z_num,z_steps,G_adj=None,p=None,show=True):
    if G_adj is None:
        #加上Alice和Bob后
        num = z_num + 2
        #Alice序号0 Bob序号N-1
        G = graph.graph(num)
        #随机给边上权值，也就是随机化的设置节点间的延时（延时没有三角不等式的限制）
        G.random_edges(100000)
    if p is None:
        p = np.ones((num,num))/(num-2)
        for i in range(num):
            p[i][0]=0
            p[i][num-1]=0
    steps = z_steps+1
    di,pi=distribution_calculate(G,p,steps)
    if show:
        if len(di)<10:
            print('测量时延的分布:')
            print(di)
            print(pi)
        #画图展示分布情况
        plt.bar(di,pi)
        plt.xticks(rotation=30)
        if len(di)>10:
            plt.xticks([])
        plt.show()
        print('密钥容量(bit):')
        print(entropy(pi))
    return entropy(pi)



