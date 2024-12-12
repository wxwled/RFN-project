# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:48:37 2021

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,GRU
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping


def time_series_pred(input_series, M, units=16, epochs=1000, show=False):
    #数据集
    max_len = min(10,len(input_series)//5)
    num_inputs = 200*len(input_series)//max_len
    dataX = []
    dataY = []

    for i in range(num_inputs):
        random_start = np.random.randint(0, len(input_series)-max_len)
        dataX.append([input_series[(random_start + i)] for i in range(max_len)])
        dataY.append(input_series[(random_start + max_len)])
    
    dataX = pad_sequences(dataX, maxlen = max_len, dtype='float32')
    X = np.reshape(dataX,(len(dataX), max_len, 1))
    y = np.array(dataY)
    
    #定义一个简单的LSTM网络
    model = Sequential()
    model.add(LSTM(units=units,input_shape=(X.shape[1],X.shape[2])))
    model.add(Dense(1,activation='sigmoid'))
    
    #定义损失函数与训练器
    model.compile(loss='MAE', optimizer='adam')
    
    #模型训练
    #禁止模型shuffle，保持整个数据的时间连续型
    #batch表示一次随机梯度下降选用的数据个数，epoch表示训练多少遍完整的数据集
    verbose = 2 if show else 0
    early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=verbose)
    model.fit(X, y, batch_size = 64, epochs= epochs, verbose= verbose, callbacks=[early_stopping])
    
    def pred_N(x, N):
        x = np.array(x)
        x = x.reshape(1,len(x),1)
        x = pad_sequences(x, maxlen = max_len, dtype='float32')
        y = []
        for i in range(N):
            new_x = model.predict(x)
            y.append(new_x[0][0])
            x = np.append(x[:,1:], new_x)
            x = x.reshape(1,len(x),1)
            x = pad_sequences(x, maxlen = max_len, dtype='float32')
        return y
    
    y_pred = pred_N(input_series[-max_len:], N = M)
    if show:
        plt.plot(range(len(input_series)),input_series)
        plt.plot(range(len(input_series),len(input_series)+M),y_pred)
        # plt.ylim(0,1)
        plt.show()
    return np.array(y_pred)



def time_series_pred_update(input_series, M, divide_rate = 3, max_step =20, units=16, epochs=500, show=False):
    x = input_series.copy()
    while len(x)<len(input_series)+M:
        nxt_M = min(max_step,len(x)//divide_rate)
        nxt = time_series_pred(x,nxt_M,units,epochs,show)
        x = np.append(x,nxt)
    return x

from scipy.signal import savgol_filter
def smooth(x, window = 5):
    return savgol_filter(x, window, 1)