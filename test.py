# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 21:01:26 2021

@author: asus
"""

from run_model import run_lstm_net,run_lstm_res
import matplotlib.pyplot as plt
from read_data import gen_mnist_data,read_train_data
import numpy as np

mnist=gen_mnist_data()

pic,label=read_train_data(mnist,10)
pic_re=np.reshape(pic,[28,28])
init_state=np.ones([1,64])
init_memory=np.ones([1,64])
state=init_state
memory=init_memory

for i in range(len(pic_re)):
    x=np.reshape(pic_re[i],[1,28])
    state,memory=run_lstm_net(x,state,memory)
 
res=run_lstm_res(state)

print(res)    
print(list(res[0]).index(max(res[0])))