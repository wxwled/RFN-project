# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:20:38 2021

@author: Administrator
"""
import numpy as np


p=np.array(
    [
        [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.50001],
        [0.333,0.333,0.335,0.337,0.339,0.34,0.341,0.342,0.342],
        [0.25,0.25,0.251,0.252,0.253,0.254,0.255,0.255,0.256],
        [0.2,0.2,0.201,0.201,0.202,0.202,0.203,0.203,0.203],
        [0.167,0.167,0.167,0.167,0.168,0.168,0.169,0.169,0.169],
        [0.143,0.143,0.143,0.143,0.143,0.143,0.144,0.144,0.144],
        [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.126],
        [0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111001]
        # [0.1  ,0.0567,0.0414,0.0337,0.0294]
      ]
    )


import matplotlib.pyplot as plt
# plt.plot(range(1,10),p.T,label=['m='+str(i) for i in range(2,10)])
# plt.legend(['m='+str(i) for i in range(2,10)],fontsize=8,loc=[0.8,0.455])

# plt.xlabel('R')
# plt.ylabel('p',rotation=0)
# # plt.ylim(0,0.6)
# plt.savefig('1.svg',format='svg')

# x = np.array([m for m in range(2,p.T.shape[0]+1)])
# for i in range(p.T.shape[0]):
#     plt.plot(x,(1/p).T[i],label='N='+str(i+1))
# plt.legend()
# plt.xlabel('m')
# plt.ylabel('1/p',rotation=0,position=(-10,0.56))
# # plt.savefig('7.svg',format='svg')
# plt.show()

# x = np.array([N for N in range(1,p.shape[0]+2)])
# for i in range(p.shape[0]):
#     plt.plot(x,1/p[i],label='m='+str(i+2))
# plt.legend()
# plt.xlabel('N')
# plt.ylabel('1/p',rotation=0,position=(-10,0.56))
# # plt.savefig('8.svg',format='svg')
# plt.show()


# H=np.array(
#     [
#       [2,2.5,2.811,3.03,3.2,3.333,3.447,3.544,3.63],
#       [3.17,4.333,5.273,6.018,6.612,7.101,7.51,7.86,8.163],
#       [4,5.664,7.129,8.4,9.483,10.415,11.226,11.94,12.573],
#       [4.644,6.691,8.565,10.267,11.788,13.145,14.361,15.458,16.455],
#       [5.17,7.523,9.725,11.778,13.666,15.395,16.979,18.436,19.728],
#       [5.615,8.221,10.697,13.04,15.235,17.283,19.19,20.969,22.634],
#       [6,8.824,11.53,14.12,16.577,18.898,21.097,23.152,25.102],
#       [6.34,9.352,12.26,15.061,17.745,20.305,22.74,25.057,27.263]
#       ]
#     )

H=np.array(
    [
      [2,3.75,5.453,7.129,8.783,10.421,12.044,13.654,15.253],
      [3.17,6.117,9.02,11.897,14.753,17.589,20.406,23.205,25.987],
      [4,7.813,11.593,15.357,19.109,22.849,26.579,30.298,34.003],
      [4.644,9.128,13.588,18.04,22.484,26.922,31.354,35.782,40.207],
      [5.17,10.201,15.215,20.223,25.227,30.230,35.232,40.233,45.234],
      [5.615,11.107,16.586,22.061,27.534,33.006,38.477,43.948,49.418],
      [6,11.891,17.771,23.644,29.511,35.373,41.231,47.084,52.933],
      [6.34,12.581,18.814,25.041,31.244,37.484,43.702,49.918,56.134]
      ]
    )


# import matplotlib.pyplot as plt
# plt.plot(range(1,10),H.T,label=['m='+str(i) for i in range(2,10)])
# plt.legend(['m='+str(i) for i in range(2,10)],fontsize=8,loc=0)

# plt.xlabel('R')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.6))
# # plt.ylim(0,0.6)
# plt.savefig('1.svg',format='svg')

# x = np.array([np.log2(m) for m in range(2,H.T.shape[0]+1)])
# for i in range(H.T.shape[0]):
#     plt.plot(x,H.T[i],label='N='+str(i+1))
# plt.legend()
# plt.xlabel('$log_2{m}$')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.56))
# plt.savefig('1.svg',format='svg')
# plt.show()

# x = np.array([np.log2(N) for N in range(1,H.shape[0]+2)])
# for i in range(H.shape[0]):
#     plt.plot(x,H[i],label='m='+str(i+2))
# plt.legend()
# plt.xlabel('$log_2{N}$')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.56))
# plt.savefig('2.svg',format='svg')
# plt.show()

p_m3=np.array([
    0.3333,0.2653,0.2367,0.2306,0.2309,0.2318,0.2331,0.2344,0.2355,0.2366,
    0.2376,0.2384,0.2393,0.24,0.241,0.2413,0.2419,0.2424,0.243,0.2433,
    0.2437,0.244,0.2443,0.2446,0.2449,0.2451,0.2453,0.2455,0.2457,0.2459,
    ])
p_m3_N40=np.array([0.247])

# plt.plot(list(range(1,31))+[40],np.hstack((1/p_m3,1/p_m3_N40)))

# # plt.plot(list(range(1,31)),1/p_m3)
# plt.xlabel('N')
# plt.ylabel('1/p',rotation=0)
# # plt.savefig('4.svg',format='svg')
# plt.show()


H_m3=np.array([
    3.17,4.333,5.273,6.018,6.613,7.101,7.51,7.86,8.163,8.43,
    8.668,8.883,9.078,9.256,9.421,9.574,9.716,9.849,9.975,10.093,
    10.205,10.312,10.413,10.51,10.603,10.692,10.777,10.859,10.938,11.014
    ])
H_m3_N40=np.array([11.656])

# import math
# f = lambda x:1.73*math.log(x,2)+2.6
# X=np.linspace(1,30,1000)
# plt.plot(list(range(1,31)),H_m3,label='H')
# plt.plot(X,[f(i) for i in X],label='appro_H')
# plt.legend()
# plt.xlabel('N')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.52))
# plt.savefig('5.svg',format='svg')
# plt.show()

H_m4=np.array([
    4,5.664,7.129,8.4,9.483,10.415,11.226,11.94,12.573,13.141,13.653,14.118,
    14.543,14.934,15.295,15.629,15.941
    ])
# x = np.array([np.log2(N) for N in range(1,len(H_m4)+1)])
# y = 3.64*x+1.074
# plt.plot(x,H_m4)
# plt.plot(x,y)
# plt.plot()
# plt.show()
# plt.plot([3.64*np.log2(N)+1.074 for N in range(1,len(H_m4)+1)],label='appro_H')
# plt.plot(H_m4,label='H')
# plt.legend()
# plt.xlabel('N')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.52))
# plt.savefig('6.svg',format='svg')
# plt.show()

p_m4=np.array([
    0.25,0.1753,0.1464,0.1349,0.1324,0.1316,0.1315,0.1317,0.1319,0.1322,0.1326,
    0.1329,0.1332,0.1336
    ])

# plt.plot(list(range(1,15)),1/p_m4)
# plt.xlabel('N')
# plt.ylabel('1/p',rotation=0)
# # plt.savefig('4.svg',format='svg')
# plt.show()

H_N2=np.array([
    2.5,4.333,5.664,6.691,7.523,8.221,8.824,9.352,9.824,10.249,10.636,10.992,
    11.32,11.625,11.911,12.178,12.43,12.668,12.894,13.109
    ])
# x = np.array([np.log2(m) for m in range(2,len(H_N2)+2)])
# y = 3.127*x-0.6258
# plt.plot(x,H_N2)
# plt.plot(x,y)
# plt.show()
# plt.plot([3.127*np.log2(m)-0.6258 for m in range(2,len(H_N2)+2)],label='appro_H')
# plt.plot(H_N2,label='H')
# plt.legend()
# plt.xlabel('N')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.52))
# plt.savefig('7.svg',format='svg')
# plt.show()

p_N2=np.array([0.5, 0.2653, 0.17526, 0.13023, 0.10348, 0.0858, 0.07327, 0.0639, 0.05669, 0.0509, 0.0462,
               0.0423, 0.039, 0.03619, 0.03375, 0.0316, 0.0297, 0.0281, 0.0265778, 0.02524, 0.024, 0.0229,
               0.02192, 0.021, 0.02015, 0.01937, 0.018649, 0.017979, 0.0173547])
# x = np.array(list(range(2,31)))
# y = 2*x-2.3505
# plt.plot(x,1/p_N2)
# plt.plot(x,y)
# plt.xlabel('m')
# plt.ylabel('1/p',rotation=0)
# # plt.savefig('4.svg',format='svg')
# plt.show()

# import math
# f = lambda x:1.73*math.log(x,2)+2.6
# X=np.linspace(1,30,1000)
# plt.plot(list(range(1,31)),H_m3,label='H')
# plt.plot(X,[f(i) for i in X],label='appro_H')
# plt.legend()
# plt.xlabel('N')
# plt.ylabel('H/bit',rotation=0,position=(-10,0.52))
# plt.savefig('5.svg',format='svg')
# plt.show()

# from scipy.optimize import curve_fit
# m_s = 4
# N_s = 5
# m = np.array(range(m_s,10))
# N = np.array(range(N_s,10))
# N, m = np.meshgrid(N, m)
# x,y = m.flatten(),N.flatten()
# z = H[m_s-2:,N_s-1:].flatten()
# X = np.vstack((x,y))

# # #X:m,N
# def Pfun(X,a,b,c,d):
#     return a*np.log2(X[0])*np.log2(X[1]) + b*np.log2(X[0]) + c*np.log2(X[1]) + d

# popt,pcov = curve_fit(Pfun,X,z)
# print(popt,pcov)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# m = np.array(range(2,10))
# N = np.array(range(1,10))
# N, m = np.meshgrid(N, m)
# Z = H[m-2,N-1]
# H_fit = popt[0]*np.log2(m)*np.log2(N) + popt[1]*np.log2(m) +popt[2]*np.log2(N) + popt[3]
# def h(a,b):
#     return popt[0]*np.log2(a)*np.log2(b) + popt[1]*np.log2(a) +popt[2]*np.log2(b) + popt[3]
# print(h(20,20))
# ax.plot_surface(N, m, H_fit, rstride = 1,   # row 行步长
#                   cstride = 2,           # colum 列步长
#                   cmap=plt.cm.autumn, zorder = 0.3 )      # 渐变颜色
# ax.plot_surface(N, m, Z, rstride = 1,   # row 行步长
#                   cstride = 2,           # colum 列步长
#                   cmap=plt.cm.winter, zorder = 0.5 )      # 渐变颜色
# plt.show()


# from scipy.optimize import curve_fit
# m_s = 4
# N_s = 5
# m = np.array(range(m_s,10))
# N = np.array(range(N_s,10))
# N, m = np.meshgrid(N, m)
# x,y = m.flatten(),N.flatten()
# z = H[m_s-2:,N_s-1:].flatten()
# X = np.vstack((x,y))

# # #X:m,N
# def Pfun(X,a,b,c,d):
#     return a*np.log2(X[0])*np.log2(X[1]) + b*np.log2(X[0]) + c*np.log2(X[1]) + d

# popt,pcov = curve_fit(Pfun,X,z)
# print(popt,pcov)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# m = np.array(range(2,100))
# N = np.array(range(1,100))
# N, m = np.meshgrid(N, m)

    
# Z = p[m-2,N-1]

# def pfuc(m,N):
#     if N==1:
#         a = m
#     if N==2 and m>2:
#         a = 2*m-2.3505
#     if N==3 and m>3:
#         a = 2.9245*m-5.0126
#     if N==4 and m>4:
#         a = 3.829*m-8.2639
#     if N==5 and m>5:
#         a = 4.6427*m-12.458
#     if N==6 and m>5:
#         a = 5.3303*m-16.126 
#     if N==7 and m>5:
#         a = 5.7018*m-17.9825
#     if N==8 and m>5:
#         a = 6.0557*m-20.0185 
#     if N==9 and m>5:
#         a = 6.3025*m-21.0084 
#     if N<=9 and m<=5:
#         return p[m-2,N-1]
#     if N > 9:
#         return pfuc(m,9)
#     return 1/a

# Z1 = np.array([[pfuc(m,N) for N in range(1,100)]for m in range(2,100)])
# # H_fit = popt[0]*np.log2(m)*np.log2(N) + popt[1]*np.log2(m) +popt[2]*np.log2(N) + popt[3]
# # def h(a,b):
# #     return popt[0]*np.log2(a)*np.log2(b) + popt[1]*np.log2(a) +popt[2]*np.log2(b) + popt[3]
# # print(h(20,20))
# # ax.plot_surface(m, N, Z1, rstride = 1,   # row 行步长
# #                   cstride = 2,           # colum 列步长
# #                   cmap=plt.cm.autumn, zorder = 0.3 )      # 渐变颜色
# ax.plot_surface(m,N, Z1, rstride = 1,   # row 行步长
#                   cstride = 2,           # colum 列步长
#                   cmap=plt.cm.autumn, zorder = 0.5 )      # 渐变颜色
# plt.show()


# x = np.array(list(range(1,51)))
# r1 = np.rint(np.random.normal(loc=-5, scale=1, size=50)).astype(int)
# r2 = np.rint(np.random.normal(loc=-5, scale=1, size=50)).astype(int)
# y0 = np.array([170.64]*49+[170.641])
# y1 = 160+r1
# y2 = 100+r2
# plt.plot(x,y0,label='Theoretical')
# plt.plot(x,y1,label='Low noise')
# plt.plot(x,y2,label='High noise')
# plt.xlabel('t/h')
# plt.ylabel('SKR/bps')
# plt.ylim(0,200)
# plt.legend()
# plt.savefig('1.svg',format='svg')
# plt.show()

x = np.array(list(range(1,51)))
r1 = np.random.normal(loc=0, scale=0.3, size=50)
# r2 = np.rint(np.random.normal(loc=-5, scale=1, size=50)).astype(int)
y0 = np.array([11.68]*49+[11.681])
y1 = 10+r1
# y2 = 100+r2
plt.plot(x,y0,label='Theoretical')
plt.plot(x,y1,label='Low noise')
# plt.plot(x,y2,label='High noise')
plt.xlabel('t/h')
plt.ylabel('SKR/bps')
plt.ylim(0,12)
plt.legend()
plt.savefig('1.svg',format='svg')
plt.show()