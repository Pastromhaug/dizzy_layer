import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.client import timeline

from data.genAdditionProblemData import genData, genEpochs
from utils.buildRNNCells import buildRNNCells

#global config variables
num_steps = 2 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 500
state_size = int(sys.argv[1])
layer_type = int(sys.argv[2])
learning_rate = float(sys.argv[3])
num_data_points = 2000
num_classes = 1
num_stacked = int(sys.argv[4])
num_test_runs = batch_size

stacked_cell = buildRNNCells(layer_type, state_size, num_stacked)

# model
x = tf.placeholder(tf.float32, [batch_size, num_steps, 2], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size], name='labels_placeholder')
# init_state = [tf.zeros([batch_size, state_size]) for i in range(num_stacked)]
init_state = stacked_cell.zero_state(batch_size, tf.float32)

inputs = tf.unpack(x, num_steps, 1)
# print(inputs)s
rnn_outputs, final_state = tf.nn.rnn(stacked_cell, inputs, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

prediction = tf.matmul(rnn_outputs[-1], W) + b
prediction = tf.squeeze(prediction)
# print("y")
# print(y)
# print("pred")
# print(prediction)
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

test_loss_summary = tf.scalar_summary('test loss layer_type: %d, state_size: %d, lr: %f, stacked: %d' % (layer_type, state_size, learning_rate, num_stacked), loss)


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

loss_summary = tf.scalar_summary('train loss layer_type: %d, state_size: %d, lr: %f, stacked: %d' % (layer_type, state_size, learning_rate, num_stacked), loss)
summary = tf.merge_summary([loss_summary])
train_writer = tf.train.SummaryWriter('./summary3', sess.graph)


def train_network(num_epochs, num_steps, state_size=4):
    # with tf.Session() as sess:


    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
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

        # print("Y:")
        # print(Y)
        # print("predictions")
        # print(prediction_)
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_add.json', 'w') as f:
        f.write(ctf)

training_losses = train_network(200,num_steps, state_size)
