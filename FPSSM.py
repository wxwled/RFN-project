# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:45:50 2020

@author: wxw
"""
import sympy

#generate a k*k symmetric symbol matrix
#symbol matrix: [1,a,b]
#               [a,1,c]
#               [b,c,1]
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

#This algorithm is used to fast calculate the power of symmetric symbol matrix
def fpssm(k,N):
    S = matrix_get('a',k)
    def exchange(f,j):
        for i in range(k):
            if not((S[i,1]==1)or(S[i,j]==1)):
                f = f.subs({S[i,1]:'x',S[i,j]:S[i,1],'x':S[i,j]})
        return f
    f,g = S[0,0],S[0,1]
    for i in range(N-1):
        w =  sympy.Matrix([f,g]+[exchange(g,j) for j in range(2,k)])
        f,g = S[:2,:]*w
    f,g = sympy.expand(f),sympy.expand(g)
    return f,g


#test
print(fpssm(3,2))