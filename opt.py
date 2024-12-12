# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:38:25 2023

@author: Administrator
"""
import sympy as sp
import re
import math
# import numpy as np

#本算法用于快速求此类矩阵的幂运算
def matrix_get(ch,k):
    A = sp.MatrixSymbol(ch,k,k)
    A = sp.Matrix(A)
    for i in range(k):
        A[i,i] = 1
        # A[i,i] = 0
    for i in range(k):
        for j in range(i,k):
            A[j,i] = A[i,j]
    return A

def fpssm(k,N):
    S = matrix_get('xq',k)
    # S = matrix_get('x',k)
    R = {
        S[j,i]:S[(j+1) % k,(i+1) % k] for j in range(0,k) for i in range(j+1,k)
        }
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
    def replace(f,r):
        l = 0
        r1={}
        r2={}
        for i in r:
            r1[i] = 't'+str(l)
            r2['t'+str(l)] = r[i]
            l += 1
        f = f.subs(r1)
        f = f.subs(r2)
        return f
    
    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        w =  sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(k-2)])
        f,g= sp.expand(S[:2,:]*w)
    w =  sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(k-2)])
    
    Sm = []
    for i in range(k):
        Sm += [w]
        w = replace(w,R)
        w = sp.Matrix([w[-1]]+w[:len(w)-1])
    
    return sp.Matrix([Sm])


def group_mat_get(ch,m):
    A = sp.zeros(m, 1)
    for i in range(1, m+1):
        ai = sp.Symbol(ch+str(i), positive=True)
        A[i-1,0] = ai
    return A

#sp.expand(A.T*fpssm(3,N)*B*B.T*fpssm(3,N)*C
def p_x(m,N,R):
    group_mats = [group_mat_get('a'+str(i),m) for i in range(R+1)]
    res = 1
    for i in range(R):
        res *= sp.expand(group_mats[i].T*fpssm(m,N)*group_mats[i+1])
    return sp.expand(res)




#optimize
x = sp.symbols('x')

def newton_step(t,f):
    return t-(eval(f)/sp.diff(eval(f),'x')).evalf(subs={'x':t})

def solve(f, x0=0.5):
    x = newton_step(x0,f)
    n = 0
    while abs(x-x0)>=1e-5:
        print('solve',n)
        x0 = x
        print(x0)
        x = newton_step(x0,f)
        n += 1
        if n > 100:
            break
    return x

def lagrange_method(H_pq,m):
    p,q = sp.symbols('p,q',positive=True)
    H_pq = -1*H_pq
    # print(H_pq)
    if m>2:
        equation = H_pq.diff(p)-H_pq.diff(q)/(m-1)
        equation = sp.expand_log(equation)
        print('diff ok')
        # print(equation)
        x = sp.symbols('x',positive=True)
        equation = equation.subs({p:x,q:1})
        # equation = sp.simplify(equation)
        # print(equation)
        # equation = sp.expand_log(equation)
        # print(equation)
        equation = str(equation)
        equation=equation.replace('log','sp.log')
        print('solve start')
        # t = sp.nsolve(eval(equation), 1)
        t = solve(equation, x0=1.1)
        # print(solve(equation, x0=0.5))
        # print(sp.nsolve(eval(equation), 0.5))
        t = t/(t+m-1)
    else:
        t = 0.5
    H_max = (-1*H_pq.subs({p:t,q:(1-t)/(m-1)}))/math.log(2)
    return t,H_max

def opt(m,N,R):
    res = str(p_x(m,N,R))[9:-3]
    res_sub = re.sub('\[.*?\]','',res)
    res_list = res_sub.split(' + ')
    print(len(res_list))
    p,q = sp.symbols('p,q',positive=True)
    H_pq = 0
    for i in res_list:
        k = i.split('*')[0]
        if k.isdigit():
            pi = eval(k)*(1/m)**R
        else:
            pi = (1/m)**R
        for j in i.split('*x')[1:]:
            pi*= eval(j)
        pi*= p**(N*R-sp.Poly(pi,q).monoms()[0][0])
        # print(pi)
        H_pq -= pi*sp.log(pi)
        # print(H_pq)
    print('H ready')
    return lagrange_method(H_pq,m)
    






