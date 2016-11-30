import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.client import timeline

from data.genAdditionProblemV1Data import genData, genEpochs, genBatch
from utils.buildRNNCells import buildRNNCells

#global config variables
num_steps = 30 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 500
state_size = int(sys.argv[1])
layer_type = int(sys.argv[2])
learning_rate = float(sys.argv[3])
num_data_points = 150000
num_stacked = int(sys.argv[4])
num_test_runs = batch_size
indices = [20,8,3]
num_classes = len(indices)+1

stacked_cell = buildRNNCells(layer_type, state_size, num_stacked)

# model
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
# init_state = [tf.zeros([batch_size, state_size]) for i in range(num_stacked)]
init_state = stacked_cell.zero_state(batch_size, tf.float32)

x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unpack(x_one_hot, axis=1)

rnn_outputs, final_state = tf.nn.rnn(stacked_cell, rnn_inputs, initial_state=init_state)

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
train_accuracy_summary = tf.scalar_summary('train acc, layer_type: %d, state_size: %d, lr: %d, stacked: %d'
                            % (layer_type, state_size, learning_rate, num_stacked),
                            accuracy)
train_loss_summary = tf.scalar_summary('train loss layer_type: %d, state_size: %d, lr: %d, stacked: %d'
                            % (layer_type, state_size, learning_rate, num_stacked),
                            total_loss)
test_accuracy_summary = tf.scalar_summary('test acc, layer_type: %d, state_size: %d, lr: %d, stacked: %d'
                            % (layer_type, state_size, learning_rate, num_stacked),
                            accuracy)
test_loss_summary = tf.scalar_summary('test loss layer_type: %d, state_size: %d, lr: %f, stacked: %d'
                            % (layer_type, state_size, learning_rate, num_stacked),
                            total_loss)

train_summaries = tf.merge_summary([train_accuracy_summary, train_loss_summary])
test_summaries = tf.merge_summary([test_accuracy_summary, test_loss_summary])
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
train_writer = tf.train.SummaryWriter('./tensorboard/additionV1', sess.graph)


# accuracies = tf.equal(tf.argmax(logits, 0), tf.argmax(y,0), 0)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# start_time = time.time()
# print("start_time1 %d" % start_time)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    # with tf.Session() as sess:


    sess.run(tf.initialize_all_variables())
    # print("---  min for  graph building ---",(time.time() - start_time)/60.0)
    # start_time = time.time()
    training_losses = []

    test_epoch = genBatch(genData(num_data_points, num_steps, batch_size, indices), batch_size, num_steps)

    for idx, epoch in enumerate(genEpochs(num_epochs, num_data_points, num_steps, batch_size, indices)):
        training_loss = 0
        train_acc = 0
        train_num_steps = 0
        training_state = [np.zeros((batch_size, state_size)) for i in range(num_stacked)]
        if verbose:
            print("EPOCH %d" % idx)
        for step, (X, Y) in enumerate(epoch):
            train_num_steps += 1

            (tr_losses, training_loss_, training_state, _, rnn_outputs_, final_state_,
            logits_, predictions_, y_as_list_, losses_, total_loss_, pred_labels_, train_accuracy_,
            correct_prediction_, train_summaries_) = \
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
                          train_summaries],
                              feed_dict={x:X, y:Y},
                              options=run_options, run_metadata=run_metadata)

            train_acc += train_accuracy_
            training_loss += training_loss_

            train_writer.add_summary(train_summaries_, idx)

        test_loss = 0
        test_acc = 0
        test_num_steps = 0
        for batch_num, (X_test, Y_test) in enumerate(test_epoch):
            (test_loss_, test_accuracy_, test_summaries_) = sess.run([total_loss, accuracy, test_summaries],
                feed_dict={x:X_test, y:Y_test},
                options=run_options, run_metadata=run_metadata)
            test_loss += test_loss_
            test_acc += test_accuracy_
            train_writer.add_summary(test_summaries_, idx)
            test_num_steps += 1

        train_acc = train_acc/train_num_steps
        training_loss = training_loss/num_steps
        test_acc = test_acc/test_num_steps
        test_loss = test_loss/test_num_steps
        if verbose:
            print("train loss: %f train acc: %f, test loss %f, test acc %f"
                    % (training_loss, train_acc, test_loss, test_acc))
        training_losses.append(training_loss)
        training_loss = 0

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('./timelines/additionV1.json', 'w') as f:
        f.write(ctf)
    return training_losses

training_losses = train_network(200,num_steps, state_size)
