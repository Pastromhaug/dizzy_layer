# from __future__ import division, print_function, absolute_import
#
# import numpy as np
# import tflearn
#
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)
# X = np.reshape(X, (-1, 28, 28))
# testX = np.reshape(testX, (-1, 28, 28))

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train = mnist.train
test = mnist.test
batchX, batchY = train.next_batch(2)
batchX_re = tf.unpack(batchX,784,1)
images = train.images
labels = train.labels

te_images = test.images
te_labels = test.labels



sess = tf.Session()
# mnist_ = sess.run(mnist)

print("mnist")
print(mnist)
print("train")
print(train)
print("batchX")
print(batchX)
print("batchY")
print(batchY)
print("images")
print(len(images))
print("labels")
print(len(labels))
print("te_images")
print(len(te_images))
print("te_labels")
print(len(te_labels))

# print("X")
# print(X_)
# print("Y")
# print(Y_)
# print("testX")
# print(testX_)
# print("testY")
# print(testY_)
