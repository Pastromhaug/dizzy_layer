import tensorflow as tf
import numpy as np

from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops

def linearTransformIdentityInit(arg, output_size, scope=None):

    with vs.variable_scope(scope or "Linear_Tansform_Identity_Init"):
        ones = tf.ones(shape=[output_size])
        identity = tf.diag(ones)
      # Now the computation.
        # with vs.variable_scope(scope or "LinearIdentity"):
        matrix = vs.get_variable(name="IdentityMatrix", dtype=tf.float32, initializer=identity)
        res = math_ops.matmul(arg, matrix)
    return res
