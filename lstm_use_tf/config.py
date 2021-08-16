# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:41:47 2021

@author: asus
"""

import string

class ModelConfig(object):
    def __init__(self):
        self.xlen=784
        self.batch_size=1
        
        self.steps=1000
        self.node_num=64
        
config=ModelConfig()

