# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:15:09 2021

@author: asus
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def gen_mnist_data():
    mnist=tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    return mnist
    

def read_train_data(mnist,num):
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    pic=x_train[num]/255.0
    label=y_train[num]
    return pic,label

def read_train_datas(mnist,num):
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    pics=[]
    labels=[]
    for i in range(num):
        sjs=np.random.randint(0,59000)
        pic=x_train[sjs]
        pic=pic/255.0
        label=y_train[sjs]
        pics.append(pic)
        labels.append(label)
    return pics,labels

def read_test_data(mnist,num):
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    pic=x_test[num]
    label=y_test[num]
    return pic,label

def trans2one_hot(num):
    if type(num)==int:
        out=np.zeros((10))
        out[num]=1
    if type(num)==list:
        out=np.zeros((len(num),10))
        for i in range(len(num)):
            out[i][num[i]]=1
    return out