import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dizzyLayer import DizzyRNNCellV1, DizzyRNNCellV2, DizzyRNNCellV3
import time
import sys

#global config variables
num_steps = 30 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 50
state_size = int(sys.argv[1])
learning_rate = 0.05
num_data_points = 15000
indeces = [3,8, 20]
num_classes = len(indeces)+1

layer_type = int(sys.argv[2])
if layer_type == 1:
    rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size)
elif layer_type == 2:
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
elif layer_type == 3:
    rnn_cell = DizzyRNNCellV1(state_size)
elif layer_type == 4:
    rnn_cell = DizzyRNNCellV2(state_size)
elif layer_type == 5:
    rnn_cell = tf.nn.rnn_cell.GRUCell(state_size)
elif layer_type == 6:
    rnn_cell = DizzyRNNCellV3(state_size)

# with tf.variable_scope('rnn_cell'):
#     W =  tf.get_variable('W', [num_classes + state_size, state_size])
#     b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0,0))
#
# def rnn_cell(rnn_input, state):
#     with tf.variable_scope('rnn_cell', reuse=True):
#         W = tf.get_variable('W', [num_classes + state_size, state_size])
#         b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
#     return tf.tanh(tf.matmul(tf.concat(1, [rnn_input, state]), W) + b)

def gen_data(size=1000000):
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

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(num_data_points), batch_size, num_steps)
# model
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unpack(x_one_hot, axis=1)

rnn_outputs, final_state = tf.nn.rnn(rnn_cell, rnn_inputs, initial_state=init_state)

[tf.histogram_summary('hidden state %d' % i, output[:,0]) for i, output in enumerate(rnn_outputs)]
# tf.histogram_summary('hidden state hist/' + type(self).__name__, output)


with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('v', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

predictions = [tf.nn.softmax(logit) for logit in logits]

y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit,label) for \
            logit, label in zip(logits, y_as_list)]

total_loss = tf.reduce_mean(losses)

pred_labels = [tf.argmax(log,1) for log in predictions]
y_as_list = tf.pack(y_as_list)
pred_labels = tf.cast(tf.pack(pred_labels), tf.int32)
# correct_prediction = [tf.equal(p,l) for p,l in zip(pred_labels, y_as_list)]
correct_prediction = tf.equal(pred_labels, y_as_list)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_accuracy = tf.scalar_summary('accuracy, layer_type: %d, state_size: %d' % (layer_type, state_size), accuracy)
train_loss = tf.scalar_summary('loss layer_type: %d, state_size: %d' % (layer_type, state_size), total_loss)

# accuracies = tf.equal(tf.argmax(logits, 0), tf.argmax(y,0), 0)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# start_time = time.time()
# print("start_time1 %d" % start_time)
sess = tf.Session()
summary = tf.merge_summary([train_accuracy, train_loss])
train_writer = tf.train.SummaryWriter('./summary2', sess.graph)
def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    # with tf.Session() as sess:


    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
    training_losses = []
    for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
        training_loss = 0
        acc = 0
        num_steps = 0
        training_state = np.zeros((batch_size, state_size))
        if verbose:
            print("EPOCH %d" % idx)
        for step, (X, Y) in enumerate(epoch):
            num_steps += 1
            # print("BATCH %d" % step)
            # print("x")
            # print(X)
            # print("y")
            # print(Y)

            (tr_losses, training_loss_, training_state, _, rnn_outputs_, final_state_,
            logits_, predictions_, y_as_list_, losses_, total_loss_, pred_labels_, accuracy_,
            correct_prediction_, summary_) = \
                sess.run([losses,
                          total_loss,
                          final_state,
                          train_step,
                          rnn_outputs,
                          final_state,
                          logits,
                          predictions,
                          y_as_list,
                          losses,
                          total_loss,
                          pred_labels,
                          accuracy,
                          correct_prediction,
                          summary],
                              feed_dict={x:X, y:Y, init_state:training_state})

            acc += accuracy
            training_loss += training_loss_

            train_writer.add_summary(summary_, idx)

        acc = acc/num_steps
        training_loss = training_loss/num_steps
        if verbose:
            print(
                  "loss:", training_loss,
                  "acc:", accuracy_)
        training_losses.append(training_loss)
        training_loss = 0

    return training_losses

training_losses = train_network(200,num_steps, state_size)
