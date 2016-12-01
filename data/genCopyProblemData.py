import numpy as np
import tensorflow as tf

def genBatch(size, num_steps, batch_size, num_classes):
    trigger = num_steps // 2
    num_batches = size // (num_steps * batch_size)
    X = np.zeros(shape=[num_batches, batch_size, num_steps, num_classes + 1])
    Y = np.zeros(shape=[num_batches, batch_size, num_steps])

    X[:, :, trigger, 0] = 1
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(trigger):
                X[i][j][k][np.random.randint(num_classes) + 1] = 1

    for i, batch in enumerate(X):
        for j, data in enumerate(batch):
            for k, step in enumerate(data):
                if k < trigger:
                    Y[i][j][k] = 0
                else:
                    Y[i][j][k] = np.argmax(X[i][j][k - trigger])

    return X, Y

def makeTestData(num_steps, num_runs, num_classes):
    trigger = num_steps // 2
    X = np.zeros(shape=[num_runs, num_steps, num_classes + 1])
    Y = np.zeros(shape=[num_runs, num_steps])

    X[:, trigger, 0] = 1
    for j in range(num_runs):
        for k in range(trigger):
            X[j][k][np.random.randint(num_classes) + 1] = 1

    for j, run in enumerate(X):
        for k, step in enumerate(run):
            if k < trigger:
                Y[j][k] = 0
            else:
                Y[j][k] = np.argmax(X[j][k - trigger])

    return X, Y

def genEpochs(num_epochs, num_data_points, num_steps, batch_size, num_classes):
    for i in range(num_epochs):
        yield genBatch(num_data_points, num_steps, batch_size, num_classes)

def genTestData(num_steps, batch_size, num_classes):
    X, Y = makeTestData(num_steps, batch_size, num_classes)

    np.save('data/copyX.npy', X)
    np.save('data/copyY.npy', Y)

def getTestData():
    return np.load('data/copyX.npy'), np.load('data/copyY.npy')
