import random
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import pandas as pd
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

image_size = 28
num_labels = 10
num_channels = 1  # grayscale


def reformat(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


train_file_path = "../input/mnist_train.csv"
test_file_path = "../input/mnist_test.csv"
data = pd.read_csv(train_file_path)
label_name = "label"
labels = data.ix[:, label_name]
dataset = data.drop(label_name, 1)
X_train, X_test, y_train, y_test = train_test_split(dataset.values, labels, test_size=0.2, random_state=0)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
X_train = reformat(X_train)
X_validation = reformat(X_validation)
X_test = reformat(X_test)
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()
y_validation = y_validation.as_matrix()
submission_dataset = pd.read_csv(test_file_path).values.reshape((-1, image_size, image_size, num_channels)).astype(
    np.float32)

print('Training set   :', X_train.shape, y_train.shape)
print('Validation set :', X_validation.shape, y_validation.shape)
print('Test set       :', X_test.shape, y_test.shape)
print('Submission data:', submission_dataset.shape)

del labels, data, dataset

# 因为照片尺寸是28*28*1，而LeNet只接收32*32*n尺寸，所以需要对mnist进行填充,这里对x_train后面两个维度填充，也就是shape[1:2]
x_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_valid = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

plt.figure(figsize=(1, 1))
index = random.randint(1, len(x_train))
# 不知道为什么要加squeeze
image = x_train[index].squeeze()
plt.imshow(image, cmap='gray')
print(f'目前显示第{index}张照片')

# 打乱数据集
x_train,y_train = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)

mu, sigma = 0, 0.1


def init_weight(shape):
    w = tf.truncated_normal(shape, mean=mu, stddev=sigma)
    return tf.Variable(w)


def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)


def LeNet(x):
    # conv1_w是conv2d的filter参数，分别为filter_height,filter_width,in_channels,out_channels
    conv1_w = init_weight((5, 5, 1, 6))
    conv1_b = init_bias(6)
    # strides=[1,1,1,1]是指图像在每一位的长度为1
    conv1 = tf.nn.conv2d(input=x, filter=conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_w = init_weight((5, 5, 6, 16))
    conv2_b = init_bias(16)
    conv2 = tf.nn.conv2d(pool_1, conv2_w, [1, 1, 1, 1], 'VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool_2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 将tensor展开为1-D的tensor，且保留batch-size，输入：[batch_size,height,width,channel]，输出：[batch_size, height * width * channel]
    fc1 = flatten(pool_2)
    fc1_w = init_weight((400, 120))
    fc1_b = init_bias(120)
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    fc2_w = init_weight((120, 84))
    fc2_b = init_bias(84)
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    fc3_w = init_weight((84, 10))
    fc3_b = init_bias(10)
    fc3 = tf.matmul(fc2, fc3_w) + fc3_b
    return fc3


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001
EPOCHS = 10
BATCH_SIZE = 128
logits = LeNet(x)
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
        x_train,y_train = shuffle(x_train,y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(x_valid, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    test_accuracy = evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
