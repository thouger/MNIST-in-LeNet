import random
from random import shuffle
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 因为照片尺寸是28*28*1，而LeNet只接收32*32*n尺寸，所以需要对mnist进行填充,这里对x_train后面两个维度填充，也就是shape[1:2]
x_train = np.pad(x_train,((0,0),(2,2),(2,2)),'constant')
x_test = np.pad(x_test,((0,0),(2,2),(2,2)),'constant')

plt.figure(figsize=(1, 1))
index = random.randint(1,len(x_train))
#不知道为什么要加squeeze
image = x_train[index].squeeze()
plt.imshow(image,cmap='gray')
print(f'目前显示第{index}张照片')

#打乱数据集
shuffle(x_train)
shuffle(x_test)

def LeNet():
