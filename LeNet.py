import random
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

from function import get_mnist_data, init_weight, init_bias

image_size = 28
num_labels = 10
num_channels = 1  # grayscale

def reformat(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


x_train, y_train, x_test, y_test, x_valid, y_valid = get_mnist_data()
# 因为照片尺寸是28*28*1，而LeNet只接收32*32*n尺寸，所以需要对mnist进行填充,这里对x_train后面两个维度填充，也就是shape[1:2]
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_valid = np.pad(x_valid, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

plt.figure(figsize=(1, 1))
index = random.randint(1, len(x_train))
# 不知道为什么要加squeeze
image = x_train[index].squeeze()
plt.imshow(image, cmap='gray')
print(f'目前显示第{index}张照片')

# 打乱数据集
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)


def LeNet(x, weights, biases):
    # conv1_w是conv2d的filter参数，分别为filter_height,filter_width,in_channels,out_channels
    # strides=[1,1,1,1]是指图像在每一位的长度为1
    conv1 = tf.nn.conv2d(input=x, filter=weights['wc1'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc1']
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = tf.nn.conv2d(pool_1, weights['wc2'], [1, 1, 1, 1], 'VALID') + biases['bc2']
    conv2 = tf.nn.relu(conv2)
    pool_2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 将tensor展开为1-D的tensor，且保留batch-size，输入：[batch_size,height,width,channel]，输出：[batch_size, height * width * channel]
    fc1 = flatten(pool_2)

    fc1 = tf.matmul(fc1, weights['wc3']) + biases['bc3']

    fc2 = tf.matmul(fc1, weights['wc4']) + biases['bc4']

    fc3 = tf.matmul(fc2, weights['wc5']) + biases['bc5']
    return fc3


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
weights = {
    'wc1': init_weight((5, 5, 1, 6)),
    'wc2': init_weight((5, 5, 6, 16)),
    'wc3': init_weight((400, 120)),
    'wc4': init_weight((120, 84)),
    'wc5': init_weight((84, 10))
}
biases = {
    'bc1': init_bias(6),
    'bc2': init_bias(16),
    'bc3': init_bias(120),
    'bc4': init_bias(84),
    'bc5': init_bias(10)
}

rate = 0.001
EPOCHS = 10
BATCH_SIZE = 128
logits = LeNet(x, weights, biases)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
loss_opertion = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_opertion)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()

    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    test_accuracy = evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
