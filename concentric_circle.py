import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# generate dataset

# generate Gaussian random data
inputs = np.random.normal(size = (10000, 2))

# drop the data to leave a gap (norm in [sqrt(2)/1.3, sqrt(2) * 1.3])
inputs_norm2 = inputs[:, 0] ** 2 + inputs[:, 1] ** 2
inputs = inputs[list(np.logical_or(inputs_norm2 > 2 * 1.3 * 1.3, 
                                   inputs_norm2 < 2 / 1.3 / 1.3)), 
                ...]
inputs_norm2 = inputs[:, 0] ** 2 + inputs[:, 1] ** 2
labels = (inputs_norm2 > 2).astype(int)


inputs_test = inputs[:128, ...]
labels_test = labels[:128, ...]
inputs = inputs[128:, ...]
labels = labels[128:, ...]

import tensorflow as tf

# build a two layer simple network
inputs_tensor = tf.placeholder(tf.float32, shape = [None, 2])
targets_tensor = tf.placeholder(tf.int64, shape = [None])

W1 = tf.get_variable('weights1', shape = [2, 4])
b1 = tf.get_variable('bias1', shape = [4])

z1 = tf.nn.elu(tf.matmul(inputs_tensor, W1) + b1)

W2 = tf.get_variable('weights2', shape = [4, 2])
b2 = tf.get_variable('bias2', shape = [2])

z2 = tf.nn.elu(tf.matmul(z1, W2) + b2)

W3 = tf.get_variable('weights3', shape = [2, 2])
b3 = tf.get_variable('bias3', shape = [2])

logits = tf.matmul(z2, W3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                              labels = tf.one_hot(targets_tensor, 2)))
pred = tf.argmax(logits, axis = -1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, 
                                           targets_tensor), 
                                  tf.float32))

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#saver.save(sess, 'tmp/concentric_model.ckpt')

saver.restore(sess, 'tmp/concentric_model.ckpt')

from margin_based_defense import margin_based_defense

# harden the network
md = margin_based_defense(logits, 
                          inputs_tensor,
                          targets_tensor,
                          loss,
                          sess,
                          other_placeholders = [],
                          update_percentage = 0.01,
                          adv_batch_size = 320,
                          one_hot = False)

md.train(inputs, labels, 
         other_placeholder_values = [],
         train_batch_size = 32,
         num_epochs = 100,
         test_inputs = inputs_test,
         test_targets = labels_test)