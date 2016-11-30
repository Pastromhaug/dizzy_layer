import numpy as np
import tensorflow as tf

def genData(size, num_steps, batch_size):
    prob = 0.025
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

def genEpochs(num_epochs, num_data_points, num_steps, batch_size):
    for i in range(num_epochs):
        yield genData(num_data_points, num_steps, batch_size)
