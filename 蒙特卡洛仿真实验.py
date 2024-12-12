# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:59:26 2020

@author: Administrator
"""
import graph
import numpy as np
import matplotlib.pyplot as plt
import collections


def entropy(f):
    s = 0
    for i in f:
        if i != 0:
            s += -i*np.log2(i)
    return s

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

def fangzheng(G ,num, steps, p, draw=0):
    N = 1000000
    x = np.array([random_travel(steps,G,p)[1] for i in range(N)])*2
    # print(x)

    p = collections.Counter(x)
    print(len(p))
    s = sum(p.values())
    for i in p:
        p[i]=p[i]/s
    p = [(i,p[i]) for i in p]
    p.sort(key=lambda x:x[0])
    print(p)
    d = [str(i[0]) for i in p]
    f = [i[1] for i in p]
    
    if draw:
        x = list(range(1,len(d)+1))
        plt.bar(x,f)
        plt.ylim(0,0.1)
        plt.xlabel('$n$')
        plt.ylabel('$p$', rotation=0)
        plt.savefig('1.svg',format='svg')
        plt.show()
                
    return entropy(f)

# def random_travel_with_noise(steps,graph,p):
#     delay = 0
#     travel_list = [0]
#     loc = np.random.choice(range(graph.num),p=p[0])
#     travel_list.append(loc)
#     delay += graph.adj[0][loc]+np.random.random()*10
#     steps -= 1
    
#     while steps:
#         nxt = np.random.choice(range(graph.num),p=p[loc])
#         travel_list.append(nxt)
#         delay += graph.adj[loc][nxt]
#         loc = nxt
#         steps -= 1
    
#     delay += graph.adj[loc][graph.num-1]
#     travel_list.append(graph.num-1)
#     return travel_list, delay

num=5
G = graph.graph(num)
G.adj = np.array([[0,2,47,7,13],
                  [2,0,17,19,23],
                  [47,17,0,29,31],
                  [7,19,29,0,37],
                  [13,23,31,37,0]])

# G.adj = np.array([[2,15,7,18,2],
#                   [15,2,17,29,15],
#                   [7,18,2,17,7],
#                   [18,29,14,2,18],
#                   [2,15,7,18,2]])

# G.adj = np.array([
#     [0,1e0,1e1,1e2,1e3],
#     [1e0,0,1e4,1e5,1e6],
#     [1e1,1e4,0,1e7,1e8],
#     [1e2,1e5,1e7,0,1e9],
#     [1e3,1e6,1e8,1e9,0]
#               ])



def strategy(p):
    pa = 1/3
    q=(1-p)/2
    return np.array([[0,pa,pa,pa,0],
                        [0,p,q,q,0],
                        [0,q,p,q,0],
                        [0,q,q,p,0]])

# num=7
# G = graph.graph(num)
# G.adj = np.array([[0,2,47,7,13,17,31],
#                   [2,0,17,19,23,34,59],
#                   [47,17,0,29,31,71,91],
#                   [7,19,29,0,37,11,29],
#                   [13,23,31,37,0,101,41],
#                   [17,34,71,11,101,0,67],
#                   [31,59,91,29,41,67,0]
#                   ])

# def strategy(p):
#     pa = 1/5
#     q=(1-p)/4
#     return np.array([
#                         [0,pa,pa,pa,pa,pa,0],
#                         [0,p,q,q,q,q,0],
#                         [0,q,p,q,q,q,0],
#                         [0,q,q,p,q,q,0],
#                         [0,q,q,q,p,q,0],
#                         [0,q,q,q,q,p,0],
#                         ])

# P1 = strategy(1/num)
# P2 = strategy(0.265)
# P3 = strategy(0.256)

steps = 2

#说明不是等概最佳
# print('仿真值：',fangzheng(G, num, steps, P1, draw=1))
# print('仿真值：',fangzheng(G, num, steps, P2, draw=1))
# print('仿真值：',fangzheng(G, num, steps, P3, draw=1))


# p1=np.linspace(0.01,0.99,99)
# H1 = [fangzheng(G, num, steps, strategy(i), draw=0) for i in p1]
# p2=[0.1,0.2,0.22,0.24,0.26,0.265,0.27,0.28,0.3,0.333,0.4,0.5,0.7,0.9]
# H2 = []
# for i in p2:
#     x = fangzheng(G, num, steps, strategy(i), draw=1)
#     print(x)
#     H2.append(x)

# plt.plot(p2,H2,'o',markersize=4,label='EXPT')
# plt.plot(p1,H1,label='THEO')
# plt.ylim(0,4.5)
# plt.xlabel('$p$')
# plt.ylabel('$H_d$   ', rotation=0)
# plt.savefig('1.svg',format='svg')
# plt.legend()
# plt.show()

# data = np.array([random_travel_with_noise(3,G,P1)[1] for i in range(10000)])
# plt.hist(data,bins=40,density=True)
# plt.xlabel('delay/s')
# plt.ylabel('prob')
# plt.savefig('1.svg',format='svg')
# plt.show()