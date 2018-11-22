# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 14:11
# @Author  : thouger
# @Email   : 1030490158@qq.com
# @File    : function.py
# @Software: PyCharm
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mu, sigma = 0, 0.1

def get_mnist_data():
    mnist = input_data.read_data_sets("../input", reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_valid, y_valid = mnist.validation.images, mnist.validation.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    del mnist
    return x_train,y_train,x_test,y_test,x_valid,y_valid

def init_weight(shape):
    w = tf.truncated_normal(shape, mean=mu, stddev=sigma)
    return tf.Variable(w)

def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)
