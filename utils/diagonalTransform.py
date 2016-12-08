import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

def diagonalTransform(X, n, std_dev=0, scope=None):
    outputs = []
    sigmas = []

    with vs.variable_scope(scope or "DiagonalTransform"):
        for t in X:
            name, x = t
            sigma = vs.get_variable(initializer=tf.random_normal([n, 1], mean=1, stddev=std_dev),
                    name="Sigmas"+name, dtype=tf.float32)
            outputs.append(sigma * x)
            sigmas.append(sigma)

    return outputs, sigmas
