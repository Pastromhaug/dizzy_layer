import tensorflow as tf

def diagonalTransform(X, n, std_dev=0):
    sigma = tf.Variable(tf.random_normal([n, 1], mean=1, stddev=std_dev))
    return sigma * X, sigma
