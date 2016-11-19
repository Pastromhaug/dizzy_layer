import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dizzyLayer import DizzyRNNCell, DizzyRNNCell2

#global config variables
num_steps = 20 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 10
state_size = 10
learning_rate = 0.1
num_data_points = 1000
indeces = [3,8]
num_classes = len(indeces)+1

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

# rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size)
# rnn_cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
rnn_cell = DizzyRNNCell(state_size)
rnn_outputs, final_state = tf.nn.rnn(rnn_cell, rnn_inputs, initial_state=init_state)

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

# accuracies = tf.equal(tf.argmax(logits, 0), tf.argmax(y,0), 0)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
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
                correct_prediction_) = \
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
                              correct_prediction],
                                  feed_dict={x:X, y:Y, init_state:training_state})

                acc += accuracy
                training_loss += training_loss_

                # print("rnn_outputs")
                # i = 0
                # for out in rnn_outputs_:
                #     print("output %d" %i)
                #     i = i + 1
                #     print(out)
                # print("")
                # print("final_state:")
                # print(final_state_)
                # print("")
                # print("Logits: ")
                # i = 0
                # for log in logits_:
                #     print("logit %d" %i)
                #     i = i + 1
                #     print(log)

                # print("")
                # print("Pred_labels: ")
                # print(pred_labels_)
                #
                # print("Correct_rediction:")
                # print(correct_prediction_)
                # i = 0
                # for log in pred_labels_:
                #     print("Pred Label %d" %i)
                #     i = i + 1
                #     print(log)
                #
                #
                #
                # print("")
                # print("Predictions:")
                # i = 0
                # for pred in predictions_:
                #     print("prediction %d" %i)
                #     i = i + 1
                #     print(pred)
                #
                # print("\ny_as_list:")
                # print(y_as_list_)
                # i = 0
                # for lst in y_as_list_:
                #     print("y_as_list %d" % i)
                #     i = i + 1
                #     print(lst)
                #
                # i = 0
                # print("\nLosses:")
                # for los in losses_:
                #     print("loss %d" %i)
                #     i = i + 1
                #     print(los)
                #
                # print("\nTotal Loss:")
                # print(total_loss_)

            acc = acc/num_steps
            training_loss = training_loss/num_steps
            if verbose:
                print(
                      "loss:", training_loss,
                      "acc:", accuracy_)
            training_losses.append(training_loss)
            training_loss = 0

    return training_losses

# data = gen_data(50)
# print(data)
# batches = gen_batch(data, 3, 5)
# for i, batch in enumerate(batches):
#     print("batch:");
#     print("x:")
#     print(batch[0])
#     print("y:")
#     print(batch[1])
training_losses = train_network(100,num_steps, state_size)
# plt.plot(training_losses)
