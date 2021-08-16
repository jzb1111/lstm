# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:51:35 2021

@author: asus
"""

import tensorflow as tf
import numpy as np
from config import config

class LSTM():
    def __init__(self,input_data,num_node=config.num_node,input_len=config.input_len,num_step=config.num_step,res_len=config.res_len):
        self.input_len=input_len
        self.num_node=num_node
        self.input_data=input_data
        self.state_size=num_node
        self.num_step=num_step
        self.kernel=self.get_kernel()
        self.bias=self.get_bias()
        self.init_state=tf.ones((config.batch_size,num_node))
        self.init_memory=tf.ones((config.batch_size,num_node))
        self.w=tf.get_variable('w',[num_node,res_len],initializer=tf.truncated_normal_initializer(-0.1, 0.1))
        self.b=tf.get_variable('b',[res_len],initializer=tf.truncated_normal_initializer(-0.1, 0.1))
        
        
    def get_kernel(self):
        return tf.get_variable("kernel",[self.input_len + self.state_size, 4 * self.num_node],initializer=tf.truncated_normal_initializer(-0.1, 0.1))
    
    def get_bias(self):
        return tf.get_variable("bias",[4 * self.num_node],initializer=tf.truncated_normal_initializer(-0.1, 0.1))
    
    def lstm_cell(self,input_x,state,memory):
        print('here',input_x,state,self.kernel)
        gate_inputs=tf.matmul(tf.concat([input_x,state],1),self.kernel)+self.bias
        print('gate_inputs',gate_inputs)
        i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)
        new_memory=memory*tf.sigmoid(f)+tf.sigmoid(i)*tf.tanh(j)
        new_state=tf.tanh(new_memory)*tf.sigmoid(o)
        return new_state,new_memory
        
    def get_res(self):
        state=self.init_state
        memory=self.init_memory
        for i in range(self.num_step):
            input_x=self.input_data[:,i]
            state,memory=self.lstm_cell(input_x,state,memory)
        res=tf.matmul(state,self.w)+self.b
        return res
    
    def model_res(self,state):
        res=tf.matmul(state,self.w)+self.b
        return tf.nn.softmax(res)