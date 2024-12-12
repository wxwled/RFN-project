# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:30:06 2020

@author: Administrator
"""
import sympy
from collections import Counter
import SeriesAnalysis

#矩阵快速幂
def fast_power(P,power):
    if power == 0:
        return 1
    if power % 2:
        return P*fast_power(P,power-1)
    else:
        return fast_power(P,power//2)**2

def matrix_get(k):
    A = sympy.MatrixSymbol('a',k,k)
    A = sympy.Matrix(A)
    for i in range(k):
        A[i,i] = 1
        # A[i,i] = 0
    for i in range(k):
        for j in range(i,k):
            A[j,i] = A[i,j]
    return A
        
def get_coefficients_cnt(e):
    return Counter((sympy.expand(e).as_coefficients_dict()).values())

def coefficient(k,N):
    A = matrix_get(k)
    # A_N = fast_power(A,N)
    A_N = A**N
    return get_coefficients_cnt(A_N[0,0]),get_coefficients_cnt(A_N[0,1])

def series_collect(N,N_series):
    series_dict_for_diag = {}
    series_dict_for_others = {}
    for k in range(2,N_series+1):
        c_d,c_o = coefficient(k,N)
        for i in c_d:
            if i in series_dict_for_diag:
                series_dict_for_diag[i].append(c_d[i])
            else:
                series_dict_for_diag[i] = [c_d[i]]
        for i in c_o:
            if i in series_dict_for_others:
                series_dict_for_others[i].append(c_o[i])
            else:
                series_dict_for_others[i] = [c_o[i]]
    return N_series,series_dict_for_diag,series_dict_for_others

def coefficient_formula(N,N_series=12):
    N_series,formula_dict_for_diag,formula_dict_for_others = series_collect(N,N_series)
    for i in formula_dict_for_diag:
        formula_dict_for_diag[i] = SeriesAnalysis.series_analysis(formula_dict_for_diag[i],N_series,'k')
        if formula_dict_for_diag[i] == None:
            print('need more series!')
            return None
    for i in formula_dict_for_others:
        formula_dict_for_others[i] = SeriesAnalysis.series_analysis(formula_dict_for_others[i],N_series,'k')
        if formula_dict_for_others[i] == None:
            print('need more series!')
            return None
    return formula_dict_for_diag,formula_dict_for_others

def coefficient_distribution(N,N_series=12):
    k = sympy.Symbol('k')
    formula_dict_for_diag,formula_dict_for_others = coefficient_formula(N,N_series)
    if formula_dict_for_diag == None:
       print('need more series!')
       return None 
    coef_distrib = {}
    for i in formula_dict_for_diag:
        if i in coef_distrib:
            coef_distrib[i] += k*formula_dict_for_diag[i]
        else:
            coef_distrib[i] = k*formula_dict_for_diag[i]
    for i in formula_dict_for_others:
        if i in coef_distrib:
            coef_distrib[i] += k*(k-1)*formula_dict_for_others[i]
        else:
            coef_distrib[i] = k*(k-1)*formula_dict_for_others[i]
    check = 0
    for i in coef_distrib:
        check += i*sympy.expand(coef_distrib[i])
    # print(check)
    return coef_distrib

def max_entropy(N,N_series=12):
    k = sympy.Symbol('k')
    coef_distrib = coefficient_distribution(N,N_series)
#    print(coef_distrib)
    delta_p = 1/k**(N+1)
    res = 0
    for i in coef_distrib:
        res += coef_distrib[i]*delta_p*i*sympy.log(1/(delta_p*i))
    return sympy.expand(res)