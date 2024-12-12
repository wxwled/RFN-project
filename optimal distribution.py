from Gradient_method import *
import sympy as sp
import re

import warnings
warnings.filterwarnings("ignore")

def mat_get(m, x):
    A = sp.zeros(m,m)
    for i in range(1, m+1):
        for j in range(1, m+1):
            pij = sp.Symbol('p_'+str(i)+str(j), positive=True)
            if i == j:
                A[i-1,j-1] = pij
            else:
                k = min(i, j)
                t = max(i, j)
                dij = sp.Symbol('d_'+str(k)+str(t), positive=True)
                A[i-1,j-1] = pij*sp.Pow(x, dij)
    return A



m = 3
#符号矩阵
x = sp.Symbol('x', positive=True)
a = sp.Wild('a')
b = sp.Wild('b')
c = sp.Wild('c')
#中间节点随机转发矩阵
S = mat_get(m, x)
#组节点随机转发向量
#A-?
s1 = sp.zeros(m, 1)
for i in range(1, m+1):
    pij = sp.Symbol('p_a'+str(i), positive=True)
    dij = sp.Symbol('d_a'+str(i), positive=True)
    s1[i-1,0] = pij*sp.Pow(x, dij)
#?-B
s2 = sp.zeros(m, 1)
for i in range(1, m+1):
    dij = sp.Symbol('d_b'+str(i), positive=True)
    s2[i-1,0] = sp.Pow(x, dij)
#B-?
s3 = sp.zeros(m, 1)
for i in range(1, m+1):
    pij = sp.Symbol('p_b'+str(i), positive=True)
    dij = sp.Symbol('d_b'+str(i), positive=True)
    s3[i-1,0] = pij*sp.Pow(x, dij)
#?-c
s4 = sp.ones(m, 1)
for i in range(1, m+1):
    dij = sp.Symbol('d_c'+str(i), positive=True)
    s4[i-1,0] = sp.Pow(x, dij)

s5 = sp.zeros(m, 1)
for i in range(1, m+1):
    pij = sp.Symbol('p_c'+str(i), positive=True)
    dij = sp.Symbol('d_c'+str(i), positive=True)
    s5[i-1,0] = pij*sp.Pow(x, dij)
t = sp.ones(m, 1)



#计算P(x)
#Example1
# 随机转发路径：A??B??C
# P = (s1.T*S*s2) * (s3.T*S*t)

#Example2
# 随机转发路径：A???B???C
P = (s1.T*S*S*s2) * (s3.T*S*S*t)

# Example3
# 随机转发路径：A?B??C???D
# P = (s1.T*s2) * (s3.T*S*s4)* (s5.T*S*t)

print('P(x) ok')

#得到d-p分布
func_symbol = P[0,0]
func_symbol = sp.expand(func_symbol)
func_symbol = sp.powsimp(func_symbol, force=True)
func_symbol = sp.collect(func_symbol, a**b)

print('d-p ok')

#得到信息熵目标函数
def gf(func_symbol):
    s = re.findall('[p0-9][\*_][_eabcp0-9+ \*]+p_[aebc0-9][0-9]', str(func_symbol))
    H = 0
    for i in s:
        t = sp.sympify(i)
        H = H + t*sp.log(t)
    return H

def sl(H):
    symbol_set = H.free_symbols
    symbol_list = list(symbol_set)
    a = []
    for i in symbol_list:
        j = str(i)
        a.append(j)
    a.sort()
    symbol_list = []
    for i in a:
        j = sp.symbols(i)
        symbol_list.append(j)
    
    return symbol_list

# func_symbol = c1(4, 6)
H = gf(func_symbol)
symbol_list = sl(H)
H = -H/sp.log(2)

print('H ok')

#消元
subs_dict = {}
remove_list = []
for i in range(m-1, len(symbol_list), m):
    subs_dict[symbol_list[i]] = 1 - sum([symbol_list[i-k-1] for k in range(m-1)])
    remove_list.append(symbol_list[i])

H = H.subs(subs_dict)
for i in remove_list:
    symbol_list.remove(i)
    
print('elements elimination ok')

#不开换元
ans = grad_up(H, symbol_list, 1/m*np.ones(len(symbol_list)))
print(ans)

# #开换元
# # 变量映射换元
# l = len(symbol_list)
# x_list = list(sp.symbols('x:'+str(l)))
# subs_dict = {}
# for i in range(l):
#     subs_dict[symbol_list[i]] = 1/(1+sp.exp(-x_list[i]))
# H = H.subs(subs_dict)
# print('elements change ok')

# ans = grad_up(H, x_list, -np.log(m-1)*np.ones(len(x_list)),step_value=1)
# print(ans)

# # 变量映射逆转换
# x = ans[0]
# for i in range(l):
#     x[i] = 1/(1+np.exp(-float(x[i][0])))
#     if (i+1)%(m-1)==0:
#         t = 1
#         for j in range(m-1):
#             t -= x[i-j]
# print(x)