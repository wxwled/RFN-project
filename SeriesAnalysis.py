# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:17:52 2020

@author: wxw
"""
#import sys
import sympy
from math import factorial

def minus_array(array):#对差运算函数
    array_len = len(array)
    array_new = []
    for i in range(array_len-1):
        differ = array[i+1]-array[i]
        if -1e-10 < differ < 1e-10:
            differ = 0
        array_new.append(differ)
    return array_new

def get_arraypic(array):#获取对差图
    arraypic = []
    array_len = len(array)
    arraypic.append(array)
    for i in range(array_len-1):
        array = minus_array(array)
        arraypic.append(array)
    return arraypic

def power_analysis(arraypic):#幂成分最高次幂分析
    zeroArrayNum = 0
    arraySum = 0
    arraypic_len = len(arraypic)
    for i in range(arraypic_len-1, -1 , -1):
        for j in range(len(arraypic[i])):
            arraySum += arraypic[i][j] ** 2
        if arraySum == 0:
            zeroArrayNum += 1
        else: break
    if zeroArrayNum != 0:
        return arraypic_len-zeroArrayNum - 1
    else:
        return -2
    
def update_array(array, coefficient, power):#更新数列，移除已知的最高次幂
    array_len = len(array)
    for i in range(array_len):
        array[i] -= coefficient * (i+1) ** power
        if -1e-10 < array[i] < 1e-10:
            array[i] = 0
    return array
    
def formula_analysis(arraypic, power):#幂成分解析
    coefficient_list = []
    power_list = []
    arraySum = 0
    for i in range(len(arraypic[0])):
        arraySum +=  arraypic[0][i] ** 2
    if arraySum == 0:
        coefficient_list.append(0)
        power_list.append(0)
    else:
        coefficient = arraypic[power][0]/factorial(power)
        coefficient_list.append(coefficient)
        power_list.append(power)
        while power > 0:
            array = update_array(arraypic[0], coefficient, power)
            arraypic = get_arraypic(array)
            power = power_analysis(arraypic)
            #调试
            #print("arraypic: ",arraypic)
            #print("power: ",power)
            #调试
            if power >= 0:
                coefficient = arraypic[power][0]/factorial(power)
                coefficient_list.append(coefficient)
                power_list.append(power)
    answer=[coefficient_list, power_list]
    return answer

def series_analysis(series,N_series=None,symbol = 'n'):
    createVar = locals()
    createVar[symbol] = sympy.Symbol(symbol)
    arraypic = get_arraypic(series)
    power = power_analysis(arraypic)
    ans = 0
    bias = 0
    if N_series:
        bias = N_series-len(series)
    #调试
    #print("arraypic: ",arraypic)
    #print("power: ",power)
    #调试
    if power == -2:
        print("I can't help you find the formula")
        print("Please check whether your data is correct or try to enter more series")
        return None
    else:
        answer = formula_analysis(arraypic, power)
        #调试
#        print("answer: ",answer)
        #调试  
        for i in range(len(answer[0])):
            if int(answer[0][i]) == answer[0][i]:
                answer[0][i] = int(answer[0][i])
            ans += answer[0][i]*(createVar[symbol]-bias)**answer[1][i]
        return ans
#test
#series = []
#length = int(input("Enter the length of series: "))
#if length >= 2:
#    print("\nEnter the value of these series")
#else:
#    print("\nThe length is illegal")
#    sys.exit(0)
#for i in range(length):
#    series.append(float(input("The %dst : "%(i+1) )))
#
#arraypic = get_arraypic(series)
#power = power_analysis(arraypic)
##调试
##print("arraypic: ",arraypic)
##print("power: ",power)
##调试
#if power == -2:
#    print("I can't help you find the formula")
#    print("Please check whether your data is correct or try to enter more series")
#else:
#    answer = formula_analysis(arraypic, power)
#    #调试
#    #print("answer: ",answer)
#    #调试  
#    for i in range(len(answer[0])):
#        if i == 0:
#            print("f(n)= %g*n^%d "%(answer[0][0],answer[1][0]), end = "")
#        else:
#            print("+ %g*n^%d "%(answer[0][i],answer[1][i]), end = "")
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    