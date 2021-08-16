# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:57:24 2021

@author: asus
"""

import tensorflow as tf
import os

def run_lstm_net(pd,st,me):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./model/lstm_net1000.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            xs=sess.graph.get_tensor_by_name("input_x:0")
            xstate=sess.graph.get_tensor_by_name("input_state:0")
            xmemory=sess.graph.get_tensor_by_name("input_memory:0")
            
            state=sess.graph.get_tensor_by_name("new_state:0")
            memory=sess.graph.get_tensor_by_name("new_memory:0")
            
            
            o_state=sess.run(state,feed_dict={xs:pd,xstate:st,xmemory:me})   
            o_memory=sess.run(memory,feed_dict={xs:pd,xstate:st,xmemory:me})
    return o_state,o_memory

def run_lstm_res(pd):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./model/lstm_res1000.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            xstate=sess.graph.get_tensor_by_name("final_state:0")
            
            res=sess.graph.get_tensor_by_name("res:0")
            
            o_res=sess.run(res,feed_dict={xstate:pd})   
            
    return o_res
