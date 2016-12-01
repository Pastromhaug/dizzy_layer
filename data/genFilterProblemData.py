import numpy as np
import tensorflow as tf

def genData(num_data_points, num_steps, batch_size, indices):
    X = np.array(np.random.choice(2, size=(num_data_points,)))
    Y = []
    for i in range(num_data_points):
        acc = 0
        for idx in indices:
            if i%num_steps >= idx:
                acc += X[i-idx]
        # Y.append(i%num_steps)
        Y.append(acc)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def genBatch(raw_data, batch_size, num_steps):
    output = []
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
        # yield (x, y)
        output.append([x,y])
    return output

def genEpochs(num_epochs, num_data_points, num_steps, batch_size, indices):
    for i in range(num_epochs):
        yield genBatch(genData(num_data_points, num_steps, batch_size, indices), batch_size, num_steps)
