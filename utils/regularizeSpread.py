import tensorflow as tf

def regularizeSpread(sigma, Lambda):
    e = tf.ones_like(sigma)
    L_sigma = 1/2 * Lambda * tf.reduce_sum((sigma - e) * (sigma - e))
    return L_sigma
