# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:25:56 2020

@author: Administrator
"""
import graph
import numpy as np
import matplotlib.pyplot as plt

# num = 4
# #Alice在0点 Bob在N-1点
# G = graph.graph(num)
# G.random_edges(20)
# G.show()
        
# steps = 8

def random_travel(steps,graph,p):
    delay = 0
    travel_list = [0]
    loc = np.random.choice(range(graph.num),p=p[0])
    travel_list.append(loc)
    delay += graph.adj[0][loc]
    steps -= 1
    
    while steps:
        nxt = np.random.choice(range(graph.num),p=p[loc])
        travel_list.append(nxt)
        delay += graph.adj[loc][nxt]
        loc = nxt
        steps -= 1
    
    delay += graph.adj[loc][graph.num-1]
    travel_list.append(graph.num-1)
    return travel_list, delay

def entropy(f):
    s = 0
    for i in f:
        if i != 0:
            s += -i*np.log2(i)
    return s

def relative_entropy(f,interval):
    return entropy(f) + np.log2(interval)
    
#num=中间节点数+Alice+Bob，steps为转发次数，P为转发策略矩阵，n_hist为离散区间个数
#draw选择要不要出分布图
def fangzheng(num, steps, p, n_hist=100, draw=0):
    G = graph.graph(num)
    G.random_edges(20)
    # G.show()
    N = 10000
    x = np.array([random_travel(steps,G,p)[1] for i in range(N)])
    f=np.histogram(x,n_hist,density=True)
    interval = f[1][1]-f[1][0]
    # print(interval)
    f = f[0]/sum(f[0])
    if draw:
        plt.hist(x,n_hist,density=True)
        plt.title(str(num)+str(' ')+str(steps)+str(' ')+str(n_hist))
        plt.show()
    # print(np.mean(x),np.var(x),entropy(f))
    return entropy(f),relative_entropy(f,interval)


num = 100
p = np.ones((num,num))/(num-2)
for i in range(num):
    p[i][0]=0
    p[i][num-1]=0
steps = np.arange(1,50,1)
entropys = np.array([fangzheng(num, i, p) for i in steps])
plt.plot(steps,entropys)
plt.show()