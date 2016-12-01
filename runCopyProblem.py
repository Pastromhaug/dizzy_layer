import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.client import timeline

from utils.buildRNNCells import buildRNNCells
from data.genCopyProblemData import genEpochs, genTestData

#global config variables
num_steps = 20 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 500
state_size = int(sys.argv[1])
layer_type = int(sys.argv[2])
learning_rate = float(sys.argv[3])
num_data_points = 10000
num_classes = 4
num_stacked = int(sys.argv[4])
num_test_runs = batch_size

stacked_cell = buildRNNCells(layer_type, state_size, num_stacked)

# model
x = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes+1], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

init_state = stacked_cell.zero_state(batch_size, tf.float32)

inputs = tf.unpack(x, num_steps, 1)
rnn_outputs, final_state = tf.nn.rnn(stacked_cell, inputs, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes + 1])
    b = tf.get_variable('b', [num_classes + 1], initializer=tf.constant_initializer(0.0))

logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
logits = tf.transpose(logits, [1, 0, 2])

predictions = tf.unpack(logits)
predictions = [tf.argmax(prediction, axis=1) for prediction in predictions]

labels = [tf.squeeze(i, squeeze_dims=[0]) for i in tf.split(0, batch_size, y)]

accuracy = [tf.equal(tf.cast(prediction, tf.int32), label) for \
        prediction, label in zip(predictions, labels)]
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label) for \
        logit, label in zip(tf.unpack(logits, batch_size), labels)]

loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

test_loss_summary = tf.scalar_summary('test loss layer_type: %d, state_size: %d' % (layer_type, state_size), loss)

sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

loss_summary = tf.scalar_summary('train loss layer_type: %d, state_size: %d' % (layer_type, state_size), loss)
summary = tf.merge_summary([loss_summary])
train_writer = tf.train.SummaryWriter('./summary3', sess.graph)


def train_network(num_epochs, num_steps, state_size=4):
    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
    training_losses = []

    X_test, Y_test = genTestData(num_steps, num_test_runs, num_classes)

    for idx, (X_epoch,Y_epoch) in enumerate(genEpochs(num_epochs, num_data_points, num_steps, batch_size, num_classes)):
        training_loss = 0
        acc = 0
        num_batches = 0
        training_state = [np.zeros((batch_size, state_size)) for i in range(num_stacked)]

        print("EPOCH %d" % idx)
        for batch in range(len(X_epoch)):
            X = X_epoch[batch]
            Y = Y_epoch[batch]

            (train_step_, loss_, summary_) = sess.run([train_step, loss, summary],
                              feed_dict={x:X, y:Y},
                              options=run_options, run_metadata=run_metadata)

            training_loss += loss_
            train_writer.add_summary(summary_, idx)
            num_batches += 1

        (test_loss, test_loss_summary_, accuracy_) = sess.run(
            [loss, test_loss_summary, accuracy],
            feed_dict={x:X_test, y:Y_test},
            options=run_options, run_metadata=run_metadata)
        train_writer.add_summary(test_loss_summary_, idx)

        training_loss = training_loss/num_batches
        print("train loss:", training_loss, "test loss:", test_loss, "test accuracy:", accuracy_)
        #  print(predictions_)
        training_loss = 0

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_add.json', 'w') as f:
        f.write(ctf)

training_losses = train_network(200, num_steps, state_size)
