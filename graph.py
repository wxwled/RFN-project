# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:18:08 2020

@author: Administrator
"""
import numpy as np
class graph:
    def __init__(self, num=0):
        self.num = num
        self.adj = np.zeros((num,num))
    
    def set_edge(self,v,w,value):
        self.adj[v][w] = value
        self.adj[w][v] = value
    
    def random_edges(self,max_delay):
        A = np.random.randint(1,max_delay,size=(self.num,self.num))
        self.adj = A + A.T
        for i in range(self.num):
            self.set_edge(i, i, 0)
    
    def show(self):
        print(self.adj)

        