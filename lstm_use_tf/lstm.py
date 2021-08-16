# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:40:14 2021

@author: asus
"""

import tensorflow as tf

from config import config

class LSTM_Cell():
    def __init__(self,train_data,train_label,num_node=64,input_size=28):
        with tf.variable_scope("f",initializer=tf.truncated_normal_initializer(-0.1,0.1)) as forget_layer:
            self.fx,self.fm,self.fb,self.fb1=self._generate_w_b(
                x_weights_size=[input_size,num_node],#此处输入数据为1个像素，与w_x相乘后得到num_node个节点
                m_weights_size=[num_node,num_node],
                biases_size=[1,num_node],
                b_size1=[1,num_node])
        with tf.variable_scope("i",initializer=tf.truncated_normal_initializer(-0.1,0.1)) as input_layer:
            self.ix,self.im,self.ib,self.ib1=self._generate_w_b(
                x_weights_size=[input_size, num_node],
                m_weights_size=[num_node, num_node],
                biases_size=[1, num_node],
                b_size1=[1,num_node])
        with tf.variable_scope("u", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as uinput_layer:
            self.cx, self.cm, self.cb,self.cb1 = self._generate_w_b(
                x_weights_size=[input_size, num_node],
                m_weights_size=[num_node, num_node],
                biases_size=[1, num_node],
                b_size1=[1,num_node])
        with tf.variable_scope("o", initializer=tf.truncated_normal_initializer(-0.1, 0.1)) as output_layer:
            self.ox, self.om, self.ob,self.ob1 = self._generate_w_b(
                x_weights_size=[input_size, num_node],
                m_weights_size=[num_node, num_node],
                biases_size=[1, num_node],
                b_size1=[1,num_node])
        self.w = tf.Variable(tf.truncated_normal([num_node, 10], -0.1, 0.1))
        self.b = tf.Variable(tf.zeros([10]))
        
        self.saved_state = tf.Variable(tf.zeros([10, num_node]), trainable=False)
        self.saved_memory = tf.Variable(tf.zeros([10, num_node]), trainable=False)

        self.train_data=train_data
        self.train_label=train_label

    def _generate_w_b(self, x_weights_size, m_weights_size, biases_size,b_size1):
        x_w = tf.get_variable("x_weights", x_weights_size)
        m_w = tf.get_variable("m_weigths", m_weights_size)
        b = tf.get_variable("biases", biases_size, initializer=tf.constant_initializer(0.0))
        b1 = tf.get_variable("biases1", biases_size, initializer=tf.constant_initializer(0.0))
        
        return x_w, m_w, b,b1

    '''def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state
    '''
    def _run(self, i, o, state):
        i=tf.reshape(i,[-1,28])
        #print('input_x',input_x)
        #in_and_state=tf.concat([input_x,state],1)
        '''forget_gate=tf.sigmoid(tf.matmul(tf.matmul(input_x,self.fx)+self.fb,self.fm)+self.fb1)
        input_gate=tf.sigmoid(tf.matmul(tf.matmul(in_and_state,self.ix)+self.ib,self.im)+self.ib1)*tf.tanh(tf.matmul(tf.matmul(in_and_state,self.ux)+self.ub,self.um)+self.ub1)
        new_mem=memory*forget_gate+input_gate
        output_gate=tf.sigmoid(tf.matmul(tf.matmul(in_and_state, self.ox)+self.ob,self.om)+self.ob1)
        new_state=output_gate*tf.tanh(new_mem)'''
        input_gate = tf.sigmoid(tf.matmul(i, self.ix) + tf.matmul(o, self.im) + self.ib)
        forget_gate = tf.sigmoid(tf.matmul(i, self.fx) + tf.matmul(o, self.fm) + self.fb)
        update = tf.matmul(i, self.cx) + tf.matmul(o, self.cm) + self.cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, self.ox) + tf.matmul(o, self.om) + self.ob)
        return output_gate * tf.tanh(state), state
    
    def loss_func(self):
        stats = []
        state = self.saved_state
        memory = self.saved_memory
        train_data=tf.reshape(self.train_data,[-1,28,28])
        #print('train',train_data[0])
        for i in range(28):#self.train_data:
            state, memory = self._run(train_data[:,i], state, memory)
            stats.append(state)
        #final_state=state
        # finnaly, the length of outputs is num_unrollings
        with tf.control_dependencies([
                self.saved_state.assign(state),
                self.saved_memory.assign(memory)
            ]):
            # concat(0, outputs) to concat the list of output on the dim 0
            # the length of outputs is batch_size
            logits= tf.sigmoid(tf.matmul(state, self.w)+ self.b)
            logits=tf.reshape(logits,[-1,10])
            train_label=tf.reshape(self.train_label,[-1,10])
            # the label should fix the size of ouputs
            loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(train_label-logits)),reduction_indices=[1]))
        #train_prediction = tf.nn.softmax(logits)
        return loss#logits, loss, train_prediction
    
    def model_get_res(self,state):
        predict=tf.sigmoid(tf.matmul(state, self.w)+self.b)
        return predict
    