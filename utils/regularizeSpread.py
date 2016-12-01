import tensorflow as tf

def regularizeSpread(sigma, Lambda):
    L_sigma = 0.5 * Lambda * tf.reduce_sum(tf.square(sigma - 1))
    return L_sigma
