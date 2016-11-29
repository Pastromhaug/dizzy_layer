import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dizzyLayer import DizzyRNNCellV1, DizzyRNNCellV2, DizzyRNNCellV3, DizzyRNNCellBottom
import time
import sys
from tensorflow.python.client import timeline



def gen_data(size, indeces, num_steps):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        acc = 0
        for idx in indeces:
            if i%num_steps >= idx:
                acc += X[i-idx]
        # Y.append(i%num_steps)
        Y.append(acc)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps, num_data_points, indeces, batch_size):
    for i in range(n):
        yield gen_batch(gen_data(num_data_points, indeces, num_steps), batch_size, num_steps)
