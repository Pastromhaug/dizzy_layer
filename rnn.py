import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dizzyLayer import DizzyRNNCellV1, DizzyRNNCellV2, DizzyRNNCellV3, DizzyRNNCellBottom
import time
import sys
from tensorflow.python.client import timeline
from gen_data import gen_epochs, gen_test_data

#global config variables
num_steps = 30 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 50
state_size = int(sys.argv[1])
layer_type = int(sys.argv[2])
learning_rate = float(sys.argv[3])
num_data_points = 15000
num_classes = 1
num_stacked = int(sys.argv[4])
num_test_runs = batch_size

if layer_type == 1:
    rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
elif layer_type == 2:
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
elif layer_type == 3:
    rnn_cell = DizzyRNNCellV1(state_size)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
        [DizzyRNNCellBottom(state_size)] + [rnn_cell] * (num_stacked-1))
elif layer_type == 4:
    rnn_cell = DizzyRNNCellV2(state_size)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
        [DizzyRNNCellBottom(state_size)] + [rnn_cell] * (num_stacked-1))
elif layer_type == 5:
    rnn_cell = tf.nn.rnn_cell.GRUCell(state_size)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_stacked)
elif layer_type == 6:
    rnn_cell = DizzyRNNCellV3(state_size)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
        [DizzyRNNCellBottom(state_size)] + [rnn_cell] * (num_stacked-1))

# model
x = tf.placeholder(tf.float32, [batch_size, num_steps, 2], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size], name='labels_placeholder')
# init_state = [tf.zeros([batch_size, state_size]) for i in range(num_stacked)]
init_state = stacked_cell.zero_state(batch_size, tf.float32)

inputs = tf.unpack(x, num_steps, 1)
rnn_outputs, final_state = tf.nn.rnn(stacked_cell, inputs, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

prediction = tf.matmul(rnn_outputs[-1], W) + b
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

# x_test = tf.placeholder(tf.float32, [num_test_runs, num_steps, 2], name='test_input_placeholder')
# y_test = tf.placeholder(tf.float32, [num_test_runs], name='test_labels_placeholder')

# test_inputs = tf.unpack(x_test, num_steps, 1)
# test_rnn_outputs, test_final_state = tf.nn.rnn(stacked_cell, test_inputs, initial_state=init_state)

# test_prediction = tf.matmul(test_rnn_outputs[-1], W) + b
# test_loss = tf.reduce_mean(tf.square(y - test_prediction))

test_loss_summary = tf.scalar_summary('test loss layer_type: %d, state_size: %d' % (layer_type, state_size), loss)


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

loss_summary = tf.scalar_summary('train loss layer_type: %d, state_size: %d' % (layer_type, state_size), loss)
summary = tf.merge_summary([loss_summary])
train_writer = tf.train.SummaryWriter('./summary3', sess.graph)


def train_network(num_epochs, num_steps, state_size=4):
    # with tf.Session() as sess:


    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
    training_losses = []

    X_test, Y_test = gen_test_data(num_steps, num_test_runs)

    for idx, (X_epoch,Y_epoch) in enumerate(gen_epochs(num_epochs, num_data_points, num_steps, batch_size)):
        training_loss = 0
        acc = 0
        num_batches = 0
        training_state = [np.zeros((batch_size, state_size)) for i in range(num_stacked)]

        print("EPOCH %d" % idx)
        for batch in range(len(X_epoch)):
            X = X_epoch[batch]
            Y = Y_epoch[batch]
            # print("x")
            # print(X)
            # print("y")
            # print(Y)
            (train_step_, loss_, summary_) = sess.run([train_step, loss, summary],
                              feed_dict={x:X, y:Y},
                              options=run_options, run_metadata=run_metadata)

            training_loss += loss_
            train_writer.add_summary(summary_, idx)
            num_batches += 1

        (test_loss, test_loss_summary_) = sess.run([loss, test_loss_summary],
            feed_dict={x:X_test, y:Y_test},
            options=run_options, run_metadata=run_metadata)
        train_writer.add_summary(test_loss_summary_, idx)

        training_loss = training_loss/num_batches
        print("train loss:", training_loss, "test loss", test_loss)
        training_loss = 0

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_add.json', 'w') as f:
        f.write(ctf)

training_losses = train_network(200,num_steps, state_size)
