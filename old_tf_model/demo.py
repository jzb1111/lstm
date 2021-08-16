# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:50:51 2021

@author: asus
"""

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
#载入数据
from tensorflow.examples.tutorials.mnist import input_data


def gen_mnist_data():
    mnist=tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    return mnist
    

def read_train_data(mnist,num):
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    pic=x_train[num]
    label=y_train[num]
    return pic,label

def read_train_datas(mnist,num):
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    pics=[]
    labels=[]
    for i in range(num):
        sjs=np.random.randint(0,59000)
        pic=x_train[sjs]/255.0
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

#mnist = input_data.read_data_sets('mnist/', one_hot=True)
#定义参数和变量
learning_rate = 0.001
batch_size = 128
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def RNN(x, n_steps, n_input, n_hidden, n_classes):
    # Parameters:
    # Input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, n_hidden]))
    # Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, n_hidden]))
    # Memory cell: input, state, and bias
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, n_hidden]))
    # Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, n_hidden]))
    # Classifier weights and biases
    w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Definition of the cell computation
    def lstm_cell(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
        state = forget_gate * state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(i, ox) +  tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state
    
    # Unrolled LSTM loop
    outputs = list()
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    output = tf.Variable(tf.zeros([batch_size, n_hidden]))
    
    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    #x = tf.transpose(x, [1, 0, 2])
    #x = tf.reshape(x, [-1, n_input])
    #x = tf.split(0, n_steps, x)
    for i in range(n_steps):#x:
        output, state = lstm_cell(x[:,i], output, state)
        outputs.append(output)
    logits =tf.matmul(outputs[-1], w) + b
    return logits

#使用RNN构建训练模型
pred = RNN(x, n_steps, n_input, n_hidden, n_classes)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(pred-y)),reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
mnist=gen_mnist_data()
#载入计算图
sess.run(init)
for step in range(20000):
    #batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x,batch_y=read_train_datas(mnist,batch_size)
    batch_y=trans2one_hot(batch_y)
    #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    if step % 50 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step) + ", Minibatch Loss= " +               "{:.6f}".format(loss) + ", Training Accuracy= " +               "{:.5f}".format(acc))
print("Optimization Finished!")

#测试数据
# Calculate accuracy for 128 mnist test images
test_len = batch_size
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))