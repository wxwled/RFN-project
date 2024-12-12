# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:45:50 2020

@author: wxw
"""
import sympy
x = sympy.Symbol('x')

#对称符号矩阵形如: [1,a,b]这样的矩阵
#               [a,1,c]
#               [b,c,1]
#本算法用于快速求此类矩阵的幂运算
def matrix_get(ch,k):
    A = sympy.MatrixSymbol(ch,k,k)
    A = sympy.Matrix(A)
    for i in range(k):
        A[i,i] = 1
        # A[i,i] = 0
    for i in range(k):
        for j in range(i,k):
            A[j,i] = A[i,j]
    return A

def fast_power(P,power):
    ans = 1
    bin_power = bin(power)[2:][::-1]
    for i in bin_power:
        if i == '1':
            ans *= P
        P = P**2
    return ans
  

def fpssm(k,N):
    S = matrix_get('a',k)
    r_e = [
            {
            S[j,1]:S[j,i] for j in range(k) if not((S[j,1]==1)or(S[j,i]==1))
                } 
            for i in range(2,k)
            ]
    def exchange(f,r):
        for i in r:
            f = f.subs({i:'x',r[i]:i,'x':r[i]})
        return f
    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        w =  sympy.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(k-2)])
        f,g=sympy.expand(S[:2,:]*w)
    return f,g

from collections import Counter
import re
def counter(k,N):
    f,g=fpssm(k,N)
    f = Counter(re.sub('\[.*?\]','',str(f)).split('+'))
    g = Counter(re.sub('\[.*?\]','',str(g)).split('+'))
    return f,g

def fpssm_plus_step(k,fg_A,fg_B):
    if fg_A==1:
        return fg_B
    if fg_B==1:
        return fg_A
    S = matrix_get('a',k)
    M = matrix_get('b',k)
    r_d1 = {S[i,j]:M[(i+1) %k,(j+1) % k] for i in range(k) for j in range(i+1,k)}
    r_d2 = {M[i,j]:S[i,j] for i in range(k) for j in range(i+1,k)}
    def R(w):
        w = w.subs(r_d1)
        w = w.subs(r_d2)
        w = [w[-1]]+w[:-1]
        return sympy.Matrix(w)
    r_e = [
            {
            S[j,1]:S[j,i] for j in range(k) if not((S[j,1]==1)or(S[j,i]==1))
                } 
            for i in range(2,k)
            ]
    def exchange(f,r):
        for i in r:
            f = f.subs({i:'x',r[i]:i,'x':r[i]})
        return f
    w_A = sympy.Matrix([fg_A[0],fg_A[1]]+[exchange(fg_A[1],r_e[j]) for j in range(k-2)])
    w_B = sympy.Matrix([fg_B[0],fg_B[1]]+[exchange(fg_B[1],r_e[j]) for j in range(k-2)])
    f = sympy.expand(w_A.T*w_B)[0]
    g = sympy.expand(R(w_A).T*w_B)[0]
    # f = (w_A.T*w_B)[0]
    # g = (R(w_A).T*w_B)[0]
    return f,g

def fpssm_plus(k,N):
    S = matrix_get('a',k)
    f,g = S[0,0],S[0,1]
    ans = 1
    bin_N = bin(N)[2:][::-1]
    for i in bin_N:
        if i == '1':
            ans = fpssm_plus_step(k,ans,(f,g))
            # print(ans)
        f,g = fpssm_plus_step(k,(f,g),(f,g))
    return ans
    

# A = matrix_get('a',100)
# import time
# start = time.time()
# sympy.expand((A**2)[0])
# end = time.time()
# print(end-start)

# start = time.time()
# fpssm(100,2)
# end = time.time()
# print(end-start)

# start = time.time()
# fpssm_plus(100,2)
# end = time.time()
# print(end-start)