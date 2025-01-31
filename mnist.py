from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


       
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class mnist():
    
    def __init__(self):
        ''' build the model and initialize variables'''
        
        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        
        # first convolution layer
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        
        # second convolution layer
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        
        # densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        
        # readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        self.logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        # loss
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        # define tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        
    def train(self):
        ''' train and store the model '''
        
        # define dataset
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                self.sess.run(self.train_step,
                              feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

                print('test accuracy %g' % self.sess.run(self.accuracy, feed_dict={
                    self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))
                
        # store the model
        self.saver.save(self.sess, "/tmp/model.ckpt")
    
    
    