# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:42:43 2021

@author: asus
"""

import tensorflow as tf
from config import config
from lstm import LSTM
from loss import get_loss
from read_data import gen_mnist_data,read_train_data,trans2one_hot,read_train_datas
import numpy as np

lr=0.01

input_x=tf.placeholder(tf.float32,[None,28,28],name='input_x_for_train')

input_label=tf.placeholder(tf.float32,[None,config.res_len])

lstm=LSTM(input_x)
res=lstm.get_res()

loss=get_loss(res,input_label)

#+for gen model
input_x_model=tf.placeholder(tf.float32,[1,28],name='input_x')
input_final_state=tf.placeholder(tf.float32,[1,64],name='final_state')
init_state=tf.placeholder(tf.float32,[1,64],name='input_state')
init_memory=tf.placeholder(tf.float32,[1,64],name='input_memory')
new_state,new_memory=lstm.lstm_cell(input_x_model,init_state,init_memory)
new_state=tf.reshape(new_state,[-1,64],name='new_state')
new_memory=tf.reshape(new_memory,[-1,64],name='new_memory')
res_model=lstm.model_res(input_final_state)
res_model=tf.reshape(res_model,[-1,10],name='res')
#-for gen model

train=tf.train.AdamOptimizer(lr).minimize(loss)
#train=tf.train.GradientDescentOptimizer(lr).minimize(loss)
init=tf.global_variables_initializer()

mnist=gen_mnist_data()

with tf.Session() as sess:
    sess.run(init)
    for i in range(500001):
        pic,label=read_train_datas(mnist,config.batch_size)
        label=trans2one_hot(label)
        pic=np.array(pic).astype(np.float32)
        sess.run(train,feed_dict={input_x:pic,input_label:label})
        l=sess.run(loss,feed_dict={input_x:pic,input_label:label})
        print(i,l)
        if i%1000==0:
            
            output_graph_def1=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['new_state','new_memory'])#sess.graph_def
            output_graph_def2=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['res'])#sess.graph_def
        
            with tf.gfile.FastGFile('./model/lstm_net'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def1.SerializeToString())
            with tf.gfile.FastGFile('./model/lstm_res'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def2.SerializeToString())