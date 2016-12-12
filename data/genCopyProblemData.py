import numpy as np
import tensorflow as tf

def genBatch(num_batches, num_steps, batch_size, num_classes, copy_len):
    trigger = num_steps - copy_len - 1
    X = np.zeros(shape=[num_batches, batch_size, num_steps, num_classes + 2])
    Y = np.ones(shape=[num_batches, batch_size, num_steps])

    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(num_steps):
                if k == trigger:
                    X[:, :, k, 0] = 1
                elif k < copy_len:
                    idx = np.random.randint(num_classes) + 2
                    X[i][j][k][idx] = 1
                    Y[i][j][trigger + k + 1] = idx
                else:
                    X[i][j][k][1] = 1

    return X, Y

def makeTestData(num_steps, num_runs, num_classes, copy_len):
    trigger = num_steps - copy_len - 1
    X = np.zeros(shape=[num_runs, num_steps, num_classes + 2])
    Y = np.ones(shape=[num_runs, num_steps])

    X[:, trigger, 0] = 1
    for j in range(num_runs):
        for k in range(num_steps):
            if k == trigger:
                X[:, k, 0] = 1
            elif k < copy_len:
                idx = np.random.randint(num_classes) + 2
                X[j][k][idx] = 1
                Y[j][trigger + k + 1] = idx
            else:
                X[j][k][1] = 1

    return X, Y

def genEpochs(num_epochs, num_batches, num_steps, batch_size, num_classes, copy_len):
    for i in range(num_epochs):
        yield genBatch(num_batches, num_steps, batch_size, num_classes, copy_len)

def genTestData(num_steps, batch_size, num_classes, copy_len):
    X, Y = makeTestData(num_steps, batch_size, num_classes, copy_len)

    np.save('data/copyX.npy', X)
    np.save('data/copyY.npy', Y)

def getTestData():
    return np.load('data/copyX.npy'), np.load('data/copyY.npy')
