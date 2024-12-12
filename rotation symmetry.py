# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:40:07 2023

@author: wxw
"""

import sympy as sp


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
    S = matrix_get('x',k)
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
        return sp.Matrix([f[-1]]+f[:len(f)-1])
    
    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        w =  sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(k-2)])
        f,g= sp.expand(S[:2,:]*w)
    w =  sp.Matrix([f,g]+[exchange(g,r_e[j]) for j in range(k-2)])
    Sm = []
    for i in range(k):
        Sm += [w]
        w = replace(w,R)
    return sp.Matrix([Sm])


def group_mat_get(ch,m):
    A = sp.zeros(m, 1)
    for i in range(1, m+1):
        ai = sp.Symbol(ch+str(i), positive=True)
        A[i-1,0] = ai
    return A


#multiply_group_mat_get ---Sm^N*B
def MG(ch,m,N):
    return sp.expand(fpssm(m,N)*group_mat_get(ch,m)*group_mat_get(ch,m).T)



def Check_R(Mat,R):
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
        return sp.Matrix([w[-1]]+w[:len(w)-1])
    return Mat[1,:] == replace(Mat[0,:],R).T

def Check_e(Mat,r_e):
    def exchange(f,r):
        for i in r:
            f = f.subs({i:'x',r[i]:i,'x':r[i]})
        return f
    return Mat[0,2] == exchange(Mat[0,1],r_e[0])

# m=3
# N=1
# res = sp.expand(MG('a1',m,N)*MG('a2',m,N)*fpssm(m,N))
# S = matrix_get('x',m)
# B = group_mat_get('a1',m)
# C = group_mat_get('a2',m)


# R = {
#         S[j,i]:S[(j+1) % m,(i+1) % m] for j in range(0,m) for i in range(j+1,m)
        
#         }
# r_e = [
#             {
#             S[j,1]:S[j,i] for j in range(m) if not((S[j,1]==1)or(S[j,i]==1))
#                 } 
#             for i in range(2,m)
#             ]
# for mat in [B,C]:
#     for i in range(m):
#         R[mat[i]] = mat[(i+1) % m]
#     for i in range(m-2):
#         r_e[i][mat[1]] = mat[1+i+1]
        
        
# print(Check_R(res,R))
# print(Check_e(res,r_e))
        
    