import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dizzyLayer import DizzyRNNCellV1, DizzyRNNCellV2, DizzyRNNCellV3, DizzyRNNCellBottom
import time
import sys
from tensorflow.python.client import timeline

prob = 0.025

def gen_data(size, num_steps, batch_size):
    num_batches = size/(num_steps * batch_size)
    X = np.random.uniform(0,1, size=[num_batches, batch_size, num_steps, 2])
    Y = np.zeros(shape=[num_batches, batch_size])
    for i, batch in enumerate(X):
        for j, data_str in enumerate(batch):
            sum = 0.0
            for k, step in enumerate(data_str):
                ran = np.random.uniform(0,1)
                if ran < prob:
                    X[i][j][k][1] = 1
                    sum += step[0]
                else:
                    X[i][j][k][1] = 0
            Y[i][j] = sum

    return X, Y

def gen_epochs(num_epochs, num_data_points, num_steps, batch_size):
    for i in range(num_epochs):
        yield gen_data(num_data_points, num_steps, batch_size)
