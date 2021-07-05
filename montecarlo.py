# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:59:26 2020

@author: Administrator
"""
import graph
import numpy as np
import matplotlib.pyplot as plt
import collections

#information entropy
def entropy(f):
    s = 0
    for i in f:
        if i != 0:
            s += -i*np.log2(i)
    return s

#random forwarding
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

#montecarlo method
def simulation(graph,steps,p,draw=0):
    N = 100000
    x = np.array([random_travel(steps,graph,p)[1] for i in range(N)])
    # print(x)
    p = collections.Counter(x)
    # print(len(p))
    s = sum(p.values())
    for i in p:
        p[i]=p[i]/s
    p = [(i,p[i]) for i in p]
    p.sort(key=lambda x:x[0])
    d = [str(i[0]) for i in p]
    f = [i[1] for i in p]
    if draw:
        plt.bar(d,f)
        # plt.ylim(0,0.1)
        plt.show()
    return entropy(f)

#example
#Set the number of the nodes(include Alice & Bob)
num=5

#Initial the RFN
G = graph.graph(num)
G.adj = np.array([[0,100,1000,7,13],
                 [100,0,17,19,23],
                 [1000,17,0,29,31],
                 [7,19,29,0,37],
                 [13,23,31,37,0]])
#the equal probability stategy
P1 = np.array([[0,1/3,1/3,1/3,0],
               [0,1/3,1/3,1/3,0],
               [0,1/3,1/3,1/3,0],
               [0,1/3,1/3,1/3,0],
               [0,1/3,1/3,1/3,0]])

#the optimal forwarding strategy
P2 = np.array([[0,1/3,1/3,1/3,0],
               [0,0.265,0.3675,0.3675,0],
               [0,0.3675,0.265,0.3675,0],
               [0,0.3675,0.3675,0.265,0],
               [0,0,0,0,0]])

#forwarding times (N+1)
steps = 3

import math
# entorpy formula for m=3,N=2
def H(p):
    q = (1-p)/2
    return -(p**2*math.log(p**2/3,2)+4*q**2*math.log(q**2/3,2)+4*p*q*math.log(2*p*q/3,2))

#draw the figure of H(p) for m=3,N=2
# p = np.linspace(0.01,0.99,1000)
# plt.plot(p,[H(i) for i in p])
# plt.plot([0.265,0.265],[0,4.333],c='b',linestyle='--')
# plt.xlabel('$p$')
# plt.ylabel('$H_d$',rotation=0,position=(-10,0.52))
# plt.ylim(-0.5,4.5)

# ax = plt.gca()
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
# plt.scatter([0.265],[0.05],s=30,c='r')
# plt.savefig('1.svg',format='svg')

print('the equal probability stategy')
print('simulation result: ',simulation(G, steps, P1, draw=1))
print('theoretical value: ',H(1/3))

print('the optimal forwarding strategy')
print('simulation result: ',simulation(G, steps, P2, draw=1))
print('theoretical value: ',H(0.265))
