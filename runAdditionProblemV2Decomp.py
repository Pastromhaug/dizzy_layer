import numpy as np
import tensorflow as tf
import sys

from data.genAdditionProblemV2Data import genData, genEpochs
from utils.buildRNNCells import buildRNNCells

num_steps = 3
batch_size = 500
state_size = int(sys.argv[1])
learning_rate = float(sys.argv[2])
Lambda = float(sys.argv[3])
num_data_points = 1500

rnn_cell = decompRNNCell(state_size)

# model
x = tf.placeholder(tf.float32, [batch_size, num_steps, 2], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size], name='labels_placeholder')
init_state = rnn_cell.zero_state(batch_size, tf.float32)

inputs = tf.unpack(x, num_steps, 1)
rnn_outputs, sigma = tf.nn.rnn(rnn_cell, inputs, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, 1])
    b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))

prediction = tf.matmul(rnn_outputs[-1], W) + b
prediction = tf.squeeze(prediction)
loss = tf.reduce_mean(tf.square(y - prediction)) + regularizeSpread(sigma, Lambda)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

test_loss_summary = tf.scalar_summary('test loss layer_type: %d, state_size: %d, lr: %f, stacked: %d' % (layer_type, state_size, learning_rate, num_stacked), loss)


sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

loss_summary = tf.scalar_summary('train loss layer_type: %d, state_size: %d, lr: %f, stacked: %d' % (layer_type, state_size, learning_rate, num_stacked), loss)
summary = tf.merge_summary([loss_summary])
train_writer = tf.train.SummaryWriter('./tensorboard/additionV2', sess.graph)


def train_network(num_epochs, num_steps, state_size=4):
    sess.run(tf.initialize_all_variables())
    training_losses = []

    (test_X_epoch,test_Y_epoch) = genData(num_data_points, num_steps, batch_size)

    for idx, (X_epoch,Y_epoch) in enumerate(genEpochs(num_epochs, num_data_points, num_steps, batch_size)):

        training_loss = 0
        num_batches = 0
        print("EPOCH %d" % idx)
        for batch in range(len(X_epoch)):
            X = X_epoch[batch]
            Y = Y_epoch[batch]
            (train_step_, loss_, summary_, prediction_) = sess.run([train_step, loss, summary, prediction],
                              feed_dict={x:X, y:Y},
                              options=run_options, run_metadata=run_metadata)

            training_loss += loss_
            train_writer.add_summary(summary_, idx)
            num_batches += 1

        test_loss = 0
        test_num_batches = 0
        for test_batch in range(len(test_X_epoch)):
            X_test = test_X_epoch[test_batch]
            Y_test = test_Y_epoch[test_batch]
            (test_loss_, test_loss_summary_) = sess.run([loss, test_loss_summary],
                feed_dict={x:X_test, y:Y_test},
                options=run_options, run_metadata=run_metadata)
            test_loss += test_loss_
            train_writer.add_summary(test_loss_summary_, idx)
            test_num_batches += 1

        test_loss = test_loss/test_num_batches
        training_loss = training_loss/num_batches
        print("train loss:", training_loss, "test loss", test_loss)
        training_loss = 0
        test_loss = 0

training_losses = train_network(200,num_steps, state_size)
