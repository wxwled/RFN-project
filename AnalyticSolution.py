# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:54:39 2021

@author: wxw
"""
import sympy as sp
import math
import re
from collections import Counter
# import SymbolReasoning as sr

x = sp.symbols('x')

def newton_step(t,f,df):
    return t-(eval(f)/eval(df)).evalf(subs={'x':t})

def solve(f, x0=0.5):
    df = str(sp.diff(eval(f),'x')).replace('log','sp.log')
    x = newton_step(x0,f,df)
    n = 0
    while abs(x-x0)>=1e-4:
        print('solve',n)
        print(x)
        x0 = x
        x = newton_step(x0,f,df)
        n += 1
        if n > 100:
            break
    return x

def matrix_get(k):
    p,q = sp.symbols('p,q',positive=True)
    A = sp.MatrixSymbol('q',k,k)
    A = sp.Matrix(A)
    for i in range(k):
        A[i,i] = p
        # A[i,i] = 0
    for i in range(k):
        for j in range(i+1,k):
            A[j,i] = A[i,j]
    return A

#复杂度O(KN)次多项式乘法运算
def fpssm(k,N):
    S = matrix_get(k)
    p,q = sp.symbols('p,q',positive=True)
    r_e = [
            {
            S[j,1]:S[j,i] for j in range(k) if not((S[j,1]==p)or(S[j,i]==p))
                } 
            for i in range(2,k)
            ]
    def exchange(f,r):
        for i in r:
            f = f.subs({i:'middle',r[i]:i,'middle':r[i]})
        return f
    # print(r_e)
    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        print('fpssm',i)
        w = sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(k-2)])
        # print(w)
        f,g=sp.expand(S[:2,:]*w)
    return f,g

def H(k,N):
    p,q = sp.symbols('p,q',positive=True)
    f,g = fpssm(k,N)
    print('fpssm ok')
    f = Counter(re.sub('\[.*?\]','',str(f)).split('+'))
    g = Counter(re.sub('\[.*?\]','',str(g)).split('+'))
    print('Counter ok')
    H_pq = 0
    for i in f:
        x = eval(i)
        H_pq -= f[i]*x*sp.log(x)
    for i in g:
        x = eval(i)
        H_pq -= (k-1)*g[i]*x*sp.log(x)
    print('H ok')
    return H_pq

def lagrange_method(k,N):
    print(k,N)
    p,q = sp.symbols('p,q',positive=True)
    H_pq = -1*H(k,N)
    print(H_pq)
    equation = (H_pq.diff(p)-H_pq.diff(q)/(k-1))/N
    print('diff ok')
    equation = sp.expand_log(equation)
    print('expand1 ok')
    x = sp.symbols('x',positive=True)
    equation = equation.subs({p:x,q:1})
    # equation = sp.simplify(equation)
    # equation = sp.expand_log(equation)
    # print(equation)
    print('expand2 ok')
    equation = str(equation)
    equation=equation.replace('log','sp.log')
    print('replace ok')
    t = solve(equation, x0=0.2457)
    # t = sp.nsolve(eval(equation), 0.1)
    # print(solve(equation, x0=0.5))
    # print(sp.nsolve(eval(equation), 0.5))
    t = t/(t+k-1)
    H_max = (-1*H_pq.subs({p:t,q:(1-t)/(k-1)}) + math.log(k))/math.log(2)
    print(t,H_max)
    return t,H_max


