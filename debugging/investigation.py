import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

# from utils.rotationPreprocess import rotationPreprocess

var = tf.Variable(tf.random_uniform([1000,1], 0, 2), name="thetas", dtype=tf.float32)
# x = tf.placeholder( shape=[2,2], name='input_placeholder', dtype=tf.float32)
x = tf.random_uniform([1000,1], 0, 2)
# mul = tf.matmul(var, x)
o = x * var
train_step = tf.train.AdagradOptimizer(0.1).minimize(o)

def run_shit():
    sess = tf.Session()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(tf.initialize_all_variables())
    train_step_ = sess.run([train_step], options=run_options, run_metadata=run_metadata,
                )#feed_dict={x: [[2,3],[5,1]]})

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('o.json', 'w') as f:
        f.write(ctf)

run_shit()
