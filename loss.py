# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:23:56 2021

@author: asus
"""

import tensorflow as tf
from config import config

def get_loss(res,label):
    res=tf.reshape(res,[-1,config.res_len])
    res=tf.nn.softmax(res)
    label=tf.reshape(label,[-1,config.res_len])
    #loss=tf.reduce_sum(tf.sqrt(tf.square(res-label)))
    loss=tf.reduce_mean(-tf.reduce_sum(label * tf.log(res),reduction_indices=[1]))
    #loss=tf.losses.mean_squared_error(label,res)
    #loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels= label,logits=res, name=None))
    return loss