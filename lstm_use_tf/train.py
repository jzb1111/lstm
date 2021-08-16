# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:26:49 2021

@author: asus
"""

import tensorflow as tf
from lstm import LSTM_Cell
from support.read_data import gen_mnist_data,read_train_data,trans2one_hot,read_train_datas
import numpy as np

lr=0.01
input_x=tf.placeholder(tf.float32,[10,28,28])

input_x_for_model=tf.placeholder(tf.float32,[1,28],name='input_x')#only for generate model
input_state_for_model=tf.placeholder(tf.float32,[1,64],name='input_state')#only for generate model
input_memory_for_model=tf.placeholder(tf.float32,[1,64],name='input_memory')#only for generate model
final_state_for_model=tf.placeholder(tf.float32,[1,64],name='final_state')#only for generate model

#input_memory=tf.placeholder(tf.float32,[1,64])
#input_status=tf.placeholder(tf.float32,[1,64])

gt_y=tf.placeholder(tf.float32,[10,10])

#+for train
lstm=LSTM_Cell(input_x,gt_y)
loss=lstm.loss_func()
#-for train

#+for gen model
model_state,model_memory=lstm._run(input_x_for_model,input_state_for_model,input_memory_for_model)
model_state=tf.reshape(model_state,[1,64],name='output_state')
model_memory=tf.reshape(model_memory,[1,64],name='output_memory')
model_res=lstm.model_get_res(final_state_for_model)
model_res=tf.reshape(model_res,[10],name='output_res')
#-for gen model

train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()
mnist=gen_mnist_data()

with tf.Session() as sess:
    sess.run(init)
    for i in range(500001):
        #sjs=np.random.randint(0,59000)
        pic,label=read_train_datas(mnist,10)
        #pic=pic/255
        pic_re=np.reshape(pic,[-1,28,28])
        label=trans2one_hot(label)
        label=np.reshape(label,[-1,10])
        sess.run(train_step,feed_dict={input_x:pic_re,gt_y:label})
        l=sess.run(loss,feed_dict={input_x:pic_re,gt_y:label})
        print(i,l)
        
        if i%1000==0:
            
            output_graph_def1=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['output_state','output_memory'])#sess.graph_def
            output_graph_def2=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['output_res'])#sess.graph_def
        
            with tf.gfile.FastGFile('./model/lstm_net'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def1.SerializeToString())
            with tf.gfile.FastGFile('./model/lstm_res'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def2.SerializeToString())