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
    pic=x_train[num]
    label=y_train[num]
    return pic,label

def read_test_data(mnist,num):
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    pic=x_test[num]
    label=y_test[num]
    return pic,label

def im2vec(im):
    vec=np.reshape(im,[-1])
    return vec

