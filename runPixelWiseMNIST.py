import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.client import timeline

from utils.buildRNNCells import buildRNNCells
from utils.regularizeSpread import regularizeSpread

num_epochs = 100
batch_size = 100
num_batches = 55000/batch_size
num_test_batches = 10000/batch_size
summary_name = sys.argv[1]
state_size = int(sys.argv[2])
layer_type = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_stacked = int(sys.argv[5])
num_test_runs = batch_size
num_classes = 10
gradient_clipping = 1.0
Lambda = 0
num_rots = state_size-1
print("layer type in pixel %d ")
if layer_type == 8:
    lambda_reg = float(sys.argv[6])

if (layer_type in [10,12,13,14,15]) and len(sys.argv) >= 7:
    num_rots = int(sys.argv[6])

if (layer_type == 12):
    lambda_reg = float(sys.argv[7])

rnn = buildRNNCells(layer_type, state_size, num_stacked, num_rots)
#--------------- Placeholders --------------------------
x = tf.placeholder(tf.float32, [batch_size, 784], name='input_placeholder')
input_data = tf.unpack(x,784,1)
input_data = [tf.reshape(j, [batch_size,1]) for j in input_data ]
# input_data = [tf.reshape(input_data[j], [batch_size,1]) for j in range(10) ]
y = tf.placeholder(tf.float32, [batch_size, 10], name='labels_placeholder')
lr = tf.placeholder(tf.float32, name='learning_rate')

#============= Build Model ==============================
init_state = rnn.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.rnn(rnn, input_data, initial_state=init_state)
sigmas = None
if layer_type == 8 or layer_type == 12:
    sigmas = rnn.get_sigmas()

#------------ Getting Loss ------------------------------
with tf.variable_scope('softmax'):
    gauss =  tf.random_normal(shape=[state_size, num_classes], mean=0.0, stddev = 1/np.sqrt(num_classes))
    W = tf.get_variable('W', initializer=gauss)
    b = tf.get_variable('v', [num_classes], initializer=tf.constant_initializer(0.0))
prediction = tf.matmul(rnn_outputs[-1], W) + b
prediction = tf.squeeze(prediction)
loss = tf.reduce_mean(tf.square(y - prediction))

#------- Singular Value Regularization  if DizzyReg------
regularization_loss = 0
if layer_type == 8 or layer_type == 12:
    regularization_loss = tf.reduce_mean([regularizeSpread(sigma, Lambda) for sigma in sigmas])

#------------------------ Optimizer ---------------------
# train_step = tf.train.AdagradOptimizer(learning_rate) \
#     .minimize(loss + regularization_loss)
optimizer = tf.train.AdagradOptimizer(learning_rate)
if layer_type in [6, 8, 10, 12, 13, 14,15]:
    print("No Gradient Clipping")
    train_step = optimizer.minimize(loss + regularization_loss)
else:
    print("Gradient Clipping: %d" % gradient_clipping)
    gradients = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -gradient_clipping, gradient_clipping), var) for grad, var in gradients]
    train_step = optimizer.apply_gradients(capped_gvs)

#--------------- Calculating Accuracy ------------------
pred_label = tf.argmax(prediction,1)
true_label = tf.argmax(y,1)
correct_prediction = tf.equal(pred_label, true_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#---------------------- Summaries ---------------------
train_accuracy_summary = tf.scalar_summary('acc_train', accuracy)
train_loss_summary = tf.scalar_summary('loss_train', loss)
test_accuracy_summary = tf.scalar_summary('acc_test', accuracy)
test_loss_summary = tf.scalar_summary('loss_test', loss)

#-------- Singular value summaries if DizzyReg---------
if layer_type == 8 or layer_type == 12:
    regularization_loss_summary = tf.scalar_summary("regularization_loss", regularization_loss)
    sigmas_summary = tf.histogram_summary("sigmas", sigmas)

#-------- Merging Summaries --------------------------
if layer_type == 8 or layer_type == 12:
    train_summaries = tf.merge_summary([train_accuracy_summary, train_loss_summary, regularization_loss_summary, sigmas_summary])
    test_summaries = tf.merge_summary([test_accuracy_summary, test_loss_summary])
else:
    train_summaries = tf.merge_summary([train_accuracy_summary, train_loss_summary])
    test_summaries = tf.merge_summary([test_accuracy_summary, test_loss_summary])
sess = tf.Session()
train_writer = tf.train.SummaryWriter('./test_shite/' + summary_name, sess.graph)



#==================== FUNCTION TO TRAIN MODEL ================
def train_network(num_epochs, state_size=4, verbose=True):
    sess.run(tf.initialize_all_variables())
    training_losses = []

    # test_epoch = genBatch(genData(num_data_points, num_steps, batch_size, indices), batch_size, num_steps)
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train = mnist.train
    test = mnist.test

    batches = []
    for i in range(num_batches):
        batches.append(train.next_batch(batch_size))

    test_batches = []
    for j in range(num_test_batches):
        test_batches.append(test.next_batch(batch_size))

    writer_count = 0
    for k in range(num_epochs):
        training_loss = 0
        train_acc = 0
        train_num_steps = 0
        print("EPOCH %d" % k)

        for i in range(num_batches):
            batch = batches[i]
            train_num_steps += 1
            if i%100 == 0:
                (training_loss_, _ , train_accuracy_, train_summaries_) = \
                    sess.run([ loss,
                              train_step,
                              accuracy,
                              train_summaries],
                                  feed_dict={x:batch[0], y:batch[1], lr:learning_rate})
                train_writer.add_summary(train_summaries_, k)
            else:
                (training_loss_, _ , train_accuracy_, ) = \
                    sess.run([ loss,
                              train_step,
                              accuracy,],
                                  feed_dict={x:batch[0], y:batch[1], lr:learning_rate})


            train_acc += train_accuracy_
            training_loss += training_loss_
            writer_count += 1

        test_loss = 0
        test_acc = 0
        test_num_steps = 0
        for j in range(num_test_batches):
            test_batch = test_batches[j]
            if j % 100 == 0:
                (test_loss_, test_accuracy_, test_summaries_) = sess.run([loss, accuracy, test_summaries],
                    feed_dict={x:test_batch[0], y:test_batch[1], lr:learning_rate})
                train_writer.add_summary(test_summaries_, k)
            else:
                (test_loss_, test_accuracy_) = sess.run([loss, accuracy],
                    feed_dict={x:test_batch[0], y:test_batch[1], lr:learning_rate})
            test_loss += test_loss_
            test_acc += test_accuracy_
            test_num_steps += 1

        train_acc = train_acc/train_num_steps
        training_loss = training_loss/train_num_steps
        test_acc = test_acc/test_num_steps
        test_loss = test_loss/test_num_steps
        print("train loss: %f train acc: %f, test loss %f, test acc %f"
                % (training_loss, train_acc, test_loss, test_acc))
        training_losses.append(training_loss)
        training_loss = 0
    return training_losses

training_losses = train_network(num_epochs, state_size)
