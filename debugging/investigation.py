import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.client import timeline

thetas = tf.Variable(tf.random_uniform([5,5], 0, 2), name="thetas", dtype=tf.float32)
o = tf.cos(thetas)

train_step = tf.train.AdagradOptimizer(0.1).minimize(o)
sess = tf.Session()
run_metadata = tf.RunMetadata()
sess.run(tf.initialize_all_variables())
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
train_step_ = sess.run([train_step], options=run_options, run_metadata=run_metadata)
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('./timelines/cos.json', 'w') as f:
    f.write(ctf)
