# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:52:11 2023

@author: wxw
"""
import sympy

def matrix_get(ch,m):
    A = sympy.MatrixSymbol(ch,m,m)
    A = sympy.Matrix(A)
    for i in range(m):
        A[i,i] = 1
        # A[i,i] = 0
    for i in range(m):
        for j in range(i,m):
            A[j,i] = A[i,j]
    return A

def group_matrix_get(ch,m):
    A = sympy.zeros(m, 1)
    for i in range(1, m+1):
        ai = sympy.Symbol(ch+str(i), positive=True)
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
    
    if type(w) == type(sympy.Matrix([1])):
        return sympy.Matrix([w[-1]]+w[:len(w)-1])
    return w

#恢复,将压缩的矩阵恢复成原型
def matrix_recovery(MG):
    f,g,r_e,r_R = MG
    m = len(r_e)+2
    w = sympy.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])#列向量
    res = [w.T]
    for i in range(m-1):
        w = replace(w,r_R)
        res += [w.T]
    return sympy.Matrix(res)
    
#快速计算Sm^N
def fpssm(m,N):
    S = matrix_get('x',m)
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
        w =  sympy.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])
        f,g=S[:2,:]*w
    return f,g,r_e,r_R

#快速计算Sm^N*bb^T
def MG(ch,m,N):
    b = group_matrix_get(ch,m)
    if N>0:
        f,g,r_e,r_R = fpssm(m,N)
    else:
        f,g,r_e,r_R = 1,0,[{} for i in range(m-2)],{}
    
    w =  sympy.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(m-2)])
    t = w.T*b
    f,g = sympy.Matrix([t*b[0],t*b[1]])
    
    for i in range(m):
        r_R[b[i]] = b[(i+1) % m]
    for i in range(m-2):
        r_e[i][b[1]] = b[i+2]
        
    return f,g,r_e,r_R



#快速生成Sm^N*cc^T，这个函数效率很高
def MG_r(ch1,ch2,MG1):
    f,g,r_e,r_R = MG1
    m = len(r_e)+2
    b = group_matrix_get(ch1,m)
    c = group_matrix_get(ch2,m)
    r = {b[i]:c[i] for i in range(m)}
    f = f.subs(r)
    g = g.subs(r)
    S = matrix_get('x',m)
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

    u = sympy.Matrix([f1,g1]+[exchange(g1,r_e1[j]) for j in range(m-2)])
    v = sympy.Matrix([f2,g2_col]+[exchange(g2_col,r_e2[j]) for j in range(m-2)])

    f = sympy.expand(u.T*v)[0]
    g = sympy.expand(u.T*replace(v,r_R2))[0]

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

#快速计算MG1*MG2:
def MG_multiply_no_expand(MG1,MG2):
    f1,g1,r_e1,r_R1 = MG1
    f2,g2,r_e2,r_R2 = MG2
    m = len(r_e1)+2

    if m>2:
        g2_col = replace(exchange(g2,r_e2[-1]),r_R2)
    else:
        g2_col = replace(g2,r_R2)
    u = sympy.Matrix([f1,g1]+[exchange(g1,r_e1[j]) for j in range(m-2)])
    v = sympy.Matrix([f2,g2_col]+[exchange(g2_col,r_e2[j]) for j in range(m-2)])
    
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

def GKFM2(m,N,R):
    MG_SmN = fpssm(m,N)
    if R == 1:
        return MG_SmN
    MG_1 = MG('a1',m,N)
    MG_all = MG_1
    for i in range(R-2):
        MG_all = MG_multiply_no_expand(MG_all,MG_r('a1','a'+str(i+2),MG_1))
    f,g,r_e,r_R = MG_multiply_no_expand(MG_all,MG_SmN)
    f = sympy.expand(f)
    g = sympy.expand(g)
    return f,g,r_e,r_R

def GKFM(m,N,R):
    MG_SmN = fpssm(m,N)
    if R == 1:
        return MG_SmN
    MG_1 = MG('a1',m,N)
    MG_all = MG_1
    for i in range(R-2):
        MG_all = MG_multiply(MG_all,MG_r('a1','a'+str(i+2),MG_1))
    return MG_multiply(MG_all,MG_SmN)
    
import time
def time_measure(f,**params):
    s = time.time()
    f(**params)
    t = time.time()
    return t-s
    
# import matplotlib.pyplot as plt
# import numpy as np
# R = np.linspace(1,10,10).astype(int)
# t1 = np.array([time_measure(GKFM,m=2,N=1,R=i) for i in R])
# t2 = np.array([time_measure(GKFM2,m=2,N=1,R=i) for i in R])
# plt.plot(R,t1,label='improved')
# plt.plot(R,t2,label='origin')
# plt.legend()
# plt.xlabel('R')
# plt.ylabel('t/s')
# plt.title('m=2,N=1')
# plt.show()


# #单矩阵验证
# fpssm_all = lambda m,N: matrix_recovery(fpssm(m,N))
# MG_fast = lambda ch,m,N: matrix_recovery(MG(ch,m,N))
# def MG_old(ch,m,N):
    # return sympy.expand(fpssm_all(m,N)*group_matrix_get(ch,m)*group_matrix_get(ch,m).T)

# print(MG_old('b',4,3)==MG_fast('b',4,3))

# #双矩阵乘积验证
# MG_bc_old = lambda m,N: sympy.expand(MG_old('b',m,N)*MG_old('c',m,N))
# MG_bc_fast = lambda m,N: matrix_recovery(MG_multiply(MG('b',m,N),MG('c',m,N)))
# print(MG_bc_old(3,2)==MG_bc_fast(3,2))

# 多矩阵乘积验证
# MG_bcd_old = lambda m,N: sympy.expand(MG_old('a0',m,N)*MG_old('a1',m,N)*MG_old('a2',m,N)*fpssm_all(m,N))
# MG_bcd_fast = lambda m,N: matrix_recovery(GKFM(m,N,4))
# print(MG_bcd_old(3,2)==MG_bcd_fast(3,2))


# #Next:快速计算MG1*MG2:
# def MG_multiply2(MG1,MG2):
#     f1,g1,r_e1,r_R1 = MG1
#     f2,g2,r_e2,r_R2 = MG2
#     m = len(r_e1)+2
    
#     r_e = []
#     for j in range(m-2):
#         e = {}
#         e.update(r_e1[j])
#         e.update(r_e2[j])
#         r_e.append(e)
    
#     r_R = {}
#     r_R.update(r_R1)
#     r_R.update(r_R2)
    
#     if m>2:
#         g2_col = replace(exchange(g2,r_e2[-1]),r_R2)
#     else:
#         g2_col = replace(g2,r_R2)
#     u = sympy.Matrix([f1,g1]+[exchange(g1,r_e1[j]) for j in range(m-2)])
#     v = sympy.Matrix([f2,g2_col]+[exchange(g2_col,r_e2[j]) for j in range(m-2)])
#     t1 = sympy.expand(g1*g2_col)
#     f = sum([sympy.expand(f1*f2),t1]+[exchange(t1,r_e[j]) for j in range(m-2)])
#     g = sympy.expand(u.T*replace(v,r_R2))[0]
    
#     return f,g,r_e,r_R   
    
    
    