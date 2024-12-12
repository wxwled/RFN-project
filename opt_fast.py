# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:17:20 2023

@author: wxw
"""
import sympy as sp
import math
import re
from collections import Counter

x = sp.symbols('x')

def newton_step(t,f):
    return t-(eval(f)/sp.diff(eval(f),'x')).evalf(subs={'x':t})

def solve(f, x0=0.5):
    x = newton_step(x0,f)
    n = 0
    while abs(x-x0)>=1e-4:
        print('solve',n)
        x0 = x
        x = newton_step(x0,f)
        n += 1
        if n > 100:
            break
    return x

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

def group_matrix_get(ch,m):
    A = sp.zeros(m, 1)
    for i in range(1, m+1):
        ai = sp.Symbol(ch+str(i), positive=True)
        A[i-1,0] = ai
    return A

#交换
def exchange(f,r):
    for i in r:
        f = f.subs({i:'x',r[i]:i,'x':r[i]})
    return f

#置换
def replace(w,r):
    l = 0
    r1={}
    r2={}
    for i in r:
        r1[i] = 't'+str(l)
        r2['t'+str(l)] = r[i]
        l += 1
    w = w.subs(r1)
    w = w.subs(r2)
    
    if type(w) == type(sp.Matrix([1])):
        return sp.Matrix([w[-1]]+w[:len(w)-1])
    return w

#恢复,将压缩的矩阵恢复成原型
def matrix_recovery(MG):
    f,g,r_e,r_R = MG
    m = len(r_e)+2
    w = sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])#列向量
    res = [w.T]
    for i in range(m-1):
        w = replace(w,r_R)
        res += [w.T]
    return sp.Matrix(res)
    
#快速计算Sm^N
def fpssm(m,N):
    S = matrix_get('xq',m)
    r_e = [
            {
            S[j,1]:S[j,i] for j in range(m) if not((S[j,1]==1)or(S[j,i]==1))
                } 
            for i in range(2,m)
            ]
    r_R = {
        S[j,i]:S[(j+1) % m,(i+1) % m] for j in range(0,m) for i in range(j+1,m)
        }

    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        w =  sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])
        f,g=sp.expand(S[:2,:]*w)
    return f,g,r_e,r_R

#快速计算Sm^N*bb^T
def MG(ch,m,N):
    b = group_matrix_get(ch,m)
    if N>0:
        f,g,r_e,r_R = fpssm(m,N)
    else:
        f,g,r_e,r_R = 1,0,[{} for i in range(m-2)],{}
    
    w =  sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])
    t = w.T*b
    f,g = sp.Matrix([t*b[0],t*b[1]])
    
    for i in range(m):
        r_R[b[i]] = b[(i+1) % m]
    for i in range(m-2):
        r_e[i][b[1]] = b[i+2]
        
    return f,g,r_e,r_R



#快速生成Sm^N*cc^T
def MG_r(ch1,ch2,MG1):
    f,g,r_e,r_R = MG1
    m = len(r_e)+2
    b = group_matrix_get(ch1,m)
    c = group_matrix_get(ch2,m)
    r = {b[i]:c[i] for i in range(m)}
    f = f.subs(r)
    g = g.subs(r)
    S = matrix_get('xq',m)
    r_e = [
            {
            S[j,1]:S[j,i] for j in range(m) if not((S[j,1]==1)or(S[j,i]==1))
                } 
            for i in range(2,m)
            ]
    r_R = {
        S[j,i]:S[(j+1) % m,(i+1) % m] for j in range(0,m) for i in range(j+1,m)
        }
    for i in range(m):
        r_R[c[i]] = c[(i+1) % m]
    for i in range(m-2):
        r_e[i][c[1]] = c[i+2]
    return f,g,r_e,r_R

#快速计算MG1*MG2:
def MG_multiply(MG1,MG2):
    f1,g1,r_e1,r_R1 = MG1
    f2,g2,r_e2,r_R2 = MG2
    m = len(r_e1)+2

    if m>2:
        g2_col = replace(exchange(g2,r_e2[-1]),r_R2)
    else:
        g2_col = replace(g2,r_R2)
    u = sp.Matrix([f1,g1]+[exchange(g1,r_e1[j]) for j in range(m-2)])
    v = sp.Matrix([f2,g2_col]+[exchange(g2_col,r_e2[j]) for j in range(m-2)])
    
    f = u.T*v
    g = u.T*replace(v,r_R2)
    
    r_e = []
    for j in range(m-2):
        e = {}
        e.update(r_e1[j])
        e.update(r_e2[j])
        r_e.append(e)
    
    r_R = {}
    r_R.update(r_R1)
    r_R.update(r_R2)
    print(len(str(f)))
    return f,g,r_e,r_R

def GKFM(m,N,R):
    MG_SmN = fpssm(m,N)
    if R == 1:
        return MG_SmN
    MG_1 = MG('a1',m,N)
    MG_all = MG_1
    for i in range(R-2):
        print(i)
        
        MG_all = MG_multiply(MG_all,MG_r('a1','a'+str(i+2),MG_1))
    return MG_multiply(MG_all,MG_SmN)

def process(f,m,N,R):
    p,q = sp.symbols('p,q',positive=True)
    f = re.sub('\[.*?\]','',str(f)).split(' + ')
    f_p = []
    for i in range(len(f)):
        k = f[i].split('*')[0]
        if k.isdigit():
            pi = eval(k)*(1/m)**R
        else:
            pi = (1/m)**R
        f[i]='*'+f[i]
        for j in f[i].split('*x')[1:]:
            pi*= eval(j)
        pi*= p**(N*R-sp.Poly(pi,q).monoms()[0][0])
        f_p.append(str(pi))
    return Counter(f_p)
    
    p,q = sp.symbols('p,q',positive=True)

def H(m,N,R):
    p,q = sp.symbols('p,q',positive=True)
    f,g = GKFM(m,N,R)[:2]
    print('GKFM ok')
    if R > 1:
        f = sp.expand(f)[0]
        g = sp.expand(g)[0]
    # print(f,g)
    print('expand ok')
    
    f = process(f,m,N,R)
    g = process(g,m,N,R)
    print(f,g)
    print('counter ok')
    H_pq = 0
    for i in f:
        x = eval(i)
        H_pq -= f[i]*x*sp.log(x)
    for i in g:
        x = eval(i)
        H_pq -= (m-1)*g[i]*x*sp.log(x)
    return H_pq*m


def lagrange_method(m,N,R):
    print(m,N,R)
    p,q = sp.symbols('p,q',positive=True)
    H_pq = -1*H(m,N,R)
    print('H ready')
    # print(H_pq)
    equation = (H_pq.diff(p)-H_pq.diff(q)/(m-1))
    equation = sp.expand_log(equation)
    print('diff ok')
    # print(equation)
    x = sp.symbols('x',positive=True)
    equation = equation.subs({p:x,q:1})
    equation = str(equation)
    equation=equation.replace('log','sp.log')
    print('solve start')
    t = solve(equation, x0=1.1)
    t = t/(t+m-1)
    H_max = (-1*H_pq.subs({p:t,q:(1-t)/(m-1)}))/math.log(2)
    print(t,H_max)
    return t,H_max