import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.client import timeline

from data.genAdditionProblemData import genData, genEpochs, getTestData
from utils.buildRNNCells import buildRNNCells
from utils.regularizeSpread import regularizeSpread

#global config variables
num_epochs = 1
num_steps = 50 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 500
summary_name = sys.argv[1]
state_size = int(sys.argv[2])
layer_type = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_data_points = 25000
num_classes = 1
num_stacked = int(sys.argv[5])
num_test_runs = batch_size
if layer_type == 8:
    Lambda = float(sys.argv[6])

rnn = buildRNNCells(layer_type, state_size, num_stacked)

# model
x = tf.placeholder(tf.float32, [batch_size, num_steps, 2], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size], name='labels_placeholder')
# init_state = [tf.zeros([batch_size, state_size]) for i in range(num_stacked)]
init_state = rnn.zero_state(batch_size, tf.float32)

inputs = tf.unpack(x, num_steps, 1)
# print(inputs)s

rnn_outputs, _ = tf.nn.rnn(rnn, inputs, initial_state=init_state)
sigmas = None
if layer_type == 8:
    sigmas = rnn.get_sigmas()

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
regularization_loss = 0
if layer_type == 8:
    regularization_loss = tf.reduce_mean([regularizeSpread(sigma, Lambda) for sigma in sigmas])
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss + regularization_loss)

test_loss_summary = tf.scalar_summary('test loss', loss)


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

loss_summary = tf.scalar_summary('train loss', loss)
if layer_type == 8:
    regularization_loss_summary = tf.scalar_summary('regularization loss', regularization_loss)
    sigmas_summary = tf.histogram_summary('sigmas', sigmas)
    summary = tf.merge_summary([loss_summary, regularization_loss_summary, sigmas_summary])
else:
    summary = tf.merge_summary([loss_summary])
train_writer = tf.train.SummaryWriter('./tensorboard/' + summary_name, sess.graph)


def train_network(num_epochs, num_steps, state_size=4):
    # with tf.Session() as sess:


    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
    training_losses = []

    # (test_X_epoch,test_Y_epoch) = genData(num_data_points, num_steps, batch_size)
    test_X_epoch,test_Y_epoch = getTestData()


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

    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format()
    # with open('./timelines/additionV2.json', 'w') as f:
    #     f.write(ctf)

training_losses = train_network(num_epochs,num_steps, state_size)
