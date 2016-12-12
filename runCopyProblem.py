import numpy as np
import tensorflow as tf
import sys
from tqdm import tqdm
from tensorflow.python.client import timeline

from utils.buildRNNCells import buildRNNCells
from utils.regularizeSpread import regularizeSpread
from data.genCopyProblemData import genEpochs, genTestData, getTestData

#global config variables
num_epochs = 200
num_steps = 100 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 100
num_batches = 10
num_classes = 10
copy_len = 10
summary_name = sys.argv[1]
state_size = int(sys.argv[2])
layer_type = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_stacked = int(sys.argv[5])
num_test_runs = batch_size
num_rots = state_size - 1
if layer_type == 8:
    lambda_reg = float(sys.argv[6])

if (layer_type == 10 or layer_type == 12) and len(sys.argv) >= 7:
    num_rots = int(sys.argv[6])

rnn = buildRNNCells(layer_type, state_size, num_stacked, num_rots)

# model
x = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes+2], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

init_state = rnn.zero_state(batch_size, tf.float32)

inputs = tf.unpack(x, num_steps, 1)
rnn_outputs, final_state = tf.nn.rnn(rnn, inputs, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes + 2])
    b = tf.get_variable('b', [num_classes + 2], initializer=tf.constant_initializer(0.0))

logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs[-copy_len:]]
logits = tf.transpose(logits, [1, 0, 2])

predictions = tf.unpack(logits)
predictions = [tf.argmax(prediction, 1) for prediction in predictions]

labels = [tf.squeeze(i, squeeze_dims=[0])[-copy_len:] for i in tf.split(0, batch_size, y)]

accuracy = [tf.equal(tf.cast(prediction, tf.int32), label) for \
        prediction, label in zip(predictions, labels)]
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label) for \
        logit, label in zip(tf.unpack(logits, batch_size), labels)]

loss = tf.reduce_mean(losses)
loss_summary = tf.scalar_summary('train loss', loss)

accuracy_summary = tf.scalar_summary('train accuracy', accuracy)

regularization_loss = 0
if layer_type == 8:
    sigmas = rnn.get_sigmas()
    regularization_loss = tf.reduce_mean([regularizeSpread(sigma, lambda_reg) for sigma in sigmas])
    regularization_loss_summary = tf.scalar_summary('regularization loss', regularization_loss)
    sigma_summary = tf.histogram_summary('sigma', sigmas)
    train_summary = tf.merge_summary([loss_summary,
                                accuracy_summary,
                                regularization_loss_summary,
                                sigma_summary])
else:
    train_summary = tf.merge_summary([loss_summary, accuracy_summary])

test_loss_summary = tf.scalar_summary('test loss', loss)
test_accuracy_summary = tf.scalar_summary('test accuracy', accuracy)

test_summary = tf.merge_summary([test_loss_summary, test_accuracy_summary])

sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

train_writer = tf.train.SummaryWriter('./tensorflaz/' + summary_name, sess.graph)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss + regularization_loss)

def train_network(num_epochs, num_steps, state_size=4):
    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
    training_losses = []

    #  X_test, Y_test = genTestData(num_steps, num_test_runs, num_classes)
    X_test, Y_test = getTestData()

    for idx, (X_epoch,Y_epoch) in enumerate(genEpochs(num_epochs, num_batches, num_steps, batch_size, num_classes, copy_len)):
        training_loss = 0
        acc = 0
        training_state = [np.zeros((batch_size, state_size)) for i in range(num_stacked)]

        print("EPOCH %d" % idx)
        for batch in tqdm(range(len(X_epoch))):
            X = X_epoch[batch]
            Y = Y_epoch[batch]

            (train_step_, loss_, train_summary_) = sess.run([train_step, loss, train_summary],
                              feed_dict={x:X, y:Y},
                              options=run_options, run_metadata=run_metadata)

            training_loss += loss_
            train_writer.add_summary(train_summary_, idx)

        (test_loss, test_summary_, accuracy_) = sess.run(
            [loss, test_summary, accuracy],
            feed_dict={x:X_test, y:Y_test},
            options=run_options, run_metadata=run_metadata)
        train_writer.add_summary(test_summary_, idx)

        training_loss = training_loss/num_batches
        print("train loss:", training_loss, "test loss:", test_loss, "test accuracy:", accuracy_)
        training_loss = 0

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_add.json', 'w') as f:
        f.write(ctf)

training_losses = train_network(num_epochs, num_steps, state_size)
