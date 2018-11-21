# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 18:45
# @Author  : thouger
# @Email   : 1030490158@qq.com
# @File    : get_mnist_data.py
# @Software: PyCharm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

train_file_path = "../input/mnist_train.csv"
test_file_path = "../input/mnist_test.csv"

image_size = 28
num_labels = 10
num_channels = 1  # grayscale

def reformat(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset

def get_mnist_data():
    data = pd.read_csv(train_file_path)
    label_name = "label"
    labels = data.ix[:, label_name]
    dataset = data.drop(label_name, 1)
    x_train, x_test, y_train, y_test = train_test_split(dataset.values, labels, test_size=0.2, random_state=0)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=0)
    x_train = reformat(x_train)
    x_validation = reformat(x_validation)
    x_test = reformat(x_test)
    y_train = y_train.as_matrix()
    y_test = y_test.as_matrix()
    y_validation = y_validation.as_matrix()

    print('Training set   :', x_train.shape, y_train.shape)
    print('Validation set :', x_validation.shape, y_validation.shape)
    print('Test set       :', x_test.shape, y_test.shape)

    del labels, data, dataset
    return x_train,y_train,x_test,y_test,x_validation,y_validation