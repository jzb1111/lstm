# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:56:55 2021

@author: asus
"""

class Config():
    def __init__(self,num_node,input_len,num_step,res_len,batch_size):
        self.num_node=num_node
        self.input_len=input_len
        self.num_step=num_step
        self.res_len=res_len
        self.batch_size=batch_size
        
config=Config(64,28,28,10,128)