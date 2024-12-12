# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:24:14 2021

@author: wxw
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM,GRU,SimpleRNN
from keras.layers import Dense,Embedding
from keras.preprocessing.sequence import pad_sequences


# #原始数据读入
# train = pd.read_csv('./data/train.csv')
# test = pd.read_csv('./data/test.csv')

#Lat:latitude Long:longitude
#地理信息+时间+病例的确诊和死亡人数

#任务给出字母序列预测下一个字母
#lstm 字母表示例
# def f(x):
#     #震荡衰减曲线
#     # return 0.5*np.exp(-0.01*x)*np.sin(0.15*x)+0.5
#     #L型曲线
#     return x/(x+1)*0.5
#     #直线的预测能力就很差
#     # return 0.002*x
import math
def f(N,i):
    if i>N-i:
        i=N-i
    x = sum([math.log(i/j+1) for j in range(1,N-i+1)])-N*math.log(2)
    return math.exp(x)

def H(N):
    return sum([-f(N,i)*math.log2(f(N,i)) for i in range(N+1)])+1
# alphabet = [f(i) for i in range(50)]
alphabet = np.array([H(i) for i in range(1,50)])

alpha = 10

alphabet = alphabet/alpha
#数据集
seq_length = 1
dataX = []
dataY = []


#出一些多字符的预测问题

for i in range(len(alphabet) - seq_length):
    dataX.append([c for c in alphabet[i : i + seq_length]])
    dataY.append(alphabet[i+seq_length])


X = np.reshape(dataX,(len(dataX), seq_length, 1))
y = np.array(dataY)

batch_size = len(y)
#定义一个简单的LSTM网络
model = Sequential()
# model.add(layers.Embedding(input_dim=1000, output_dim=64))
# model.add(GRU(units=8,input_shape=(X.shape[1],X.shape[2])))
model.add(LSTM(units=16,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(LSTM(units=8))
# model.add(LSTM(units=8))
# model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#定义损失函数与训练器
model.compile(loss='MAE', optimizer='adam')

#模型训练
#禁止模型shuffle，保持整个数据的时间连续型
#batch表示一次随机梯度下降选用的数据个数，epoch表示训练多少遍完整的数据集
history = model.fit(X, y, batch_size = batch_size, epochs= 5000, shuffle=False, verbose=2)

# plt.plot(history.history['loss'])
# plt.show()

#模型测试
def pred(x, f=model):
    x = np.array(x)
    x = np.reshape(x,(len(x), seq_length, 1))
    return model.predict(x)[0][0]

def pred_N(x, f=model, N=200):
    x = np.array(x)
    x = np.reshape(x,(len(x), seq_length, 1))
    y = [X[0][0][0]]
    for i in range(N):
        new_x = model.predict(x)
        y.append(new_x[0][0])
        x = np.array([new_x])
        x = np.reshape(x,(len(x), seq_length, 1))
    return np.array(y)

N = 100
y_pred = pred_N(X[:1], N = N)
plt.plot(y_pred*alpha)
plt.plot(alphabet*alpha,c='red')
plt.plot([H(i) for i in range(1,N+1)], linewidth = 0.5)
plt.show()


    

