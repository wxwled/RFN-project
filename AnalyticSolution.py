# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:54:39 2021

@author: wxw
"""
import sympy as sp
import math
import re
from collections import Counter

x = sp.symbols('x')
#Newton Method for solve an equation
def newton_step(t,f,df):
    return t-(eval(f)/eval(df)).evalf(subs={'x':t})

def solve(f, x0=0.5):
    df = str(sp.diff(eval(f),'x')).replace('log','sp.log')
    x = newton_step(x0,f,df)
    n = 0
    while abs(x-x0)>=1e-4:
        # print('solve',n)
        # print(x)
        x0 = x
        x = newton_step(x0,f,df)
        n += 1
        if n > 100:
            break
    return x

def matrix_get(m):
    p,q = sp.symbols('p,q',positive=True)
    A = sp.MatrixSymbol('q',m,m)
    A = sp.Matrix(A)
    for i in range(m):
        A[i,i] = p
        # A[i,i] = 0
    for i in range(m):
        for j in range(i+1,m):
            A[j,i] = A[i,j]
    return A

#Algorithm for fast calculating the power of symmetric symbol matrix
def fpssm(m,N):
    S = matrix_get(m)
    p,q = sp.symbols('p,q',positive=True)
    r_e = [
            {
            S[j,1]:S[j,i] for j in range(m) if not((S[j,1]==p)or(S[j,i]==p))
                } 
            for i in range(2,m)
            ]
    def exchange(f,r):
        for i in r:
            f = f.subs({i:'middle',r[i]:i,'middle':r[i]})
        return f
    # print(r_e)
    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        # print('fpssm',i)
        w = sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])
        # print(w)
        f,g=sp.expand(S[:2,:]*w)
    return f,g

#Get the expression of H(m,N)
def H(m,N):
    p,q = sp.symbols('p,q',positive=True)
    f,g = fpssm(m,N)
    # print('fpssm ok')
    f = Counter(re.sub('\[.*?\]','',str(f)).split('+'))
    g = Counter(re.sub('\[.*?\]','',str(g)).split('+'))
    # print('Counter ok')
    H_pq = 0
    for i in f:
        x = eval(i)
        H_pq -= f[i]*x*sp.log(x)
    for i in g:
        x = eval(i)
        H_pq -= (m-1)*g[i]*x*sp.log(x)
    # print('H ok')
    return H_pq

#lagrange multiplier method for maximum H(m,N)
def lagrange_method(m,N):
    # print(m,N)
    p,q = sp.symbols('p,q',positive=True)
    H_pq = -1*H(m,N)
    # print(H_pq)
    equation = (H_pq.diff(p)-H_pq.diff(q)/(m-1))/N
    # print('diff ok')
    equation = sp.expand_log(equation)
    # print('expand1 ok')
    x = sp.symbols('x',positive=True)
    equation = equation.subs({p:x,q:1})
    # equation = sp.simplify(equation)
    # equation = sp.expand_log(equation)
    # print(equation)
    # print('expand2 ok')
    equation = str(equation)
    equation=equation.replace('log','sp.log')
    # print('replace ok')
    t = solve(equation, x0=0.2457)
    # t = sp.nsolve(eval(equation), 0.1)
    # print(solve(equation, x0=0.5))
    # print(sp.nsolve(eval(equation), 0.5))
    t = t/(t+m-1)
    H_max = (-1*H_pq.subs({p:t,q:(1-t)/(m-1)}) + math.log(m))/math.log(2)
    # print(t,H_max)
    return t,H_max

#example
p,H_max = lagrange_method(3,2)
print('The optimal forwarding strategy for m=3 N=2 is')
print('p=',p)
print('The maximun entropy of end-to-end delay for m=3 N=2 is')
print('H_max=',H_max)