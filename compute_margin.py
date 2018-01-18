import tensorflow as tf
import numpy as np

class margin():
    def __init__(self, logits, inputs_tensor, targets_tensor,
                 sess,
                 other_placeholders = []):
        '''
        build the computation graph for distance computation
        
        Args:
        logits - the logit tensor in a network
        inputs_tensor - the input tensor in a network
        targets_tensor - the target tensor in a network
        sess - tensorflow session
        other_placeholders - a list of other placeholders in the network
        '''
        
        # determine the number of classes
        self.num_classes = logits.get_shape().as_list()[-1]
        
        # define a trainable parameter
        self.closest = tf.get_variable('closest', 
                                       shape = [1] + inputs_tensor.get_shape().as_list()[1:])
        
        # placeholders
        self.logits = logits
        self.inputs_tensor = inputs_tensor
        self.targets_tensor = targets_tensor
        self.other_placeholders = other_placeholders
        
        # create optimizer & graident placeholder
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.gradients = tf.placeholder(shape = inputs_tensor.get_shape(),
                                        dtype = tf.float32)
        self.compute_gradients = [(self.gradients, self.closest)]
        self.apply_gradients = self.optimizer.apply_gradients(self.compute_gradients)
        
        # define logistic gradients
        self.l_grad = [0] * self.num_classes
        for c in xrange(self.num_classes):
            self.l_grad[c] = tf.gradients(self.logits[0, c],
                                          self.inputs_tensor)[0]
        # define distance gradients
        self.original_inputs = tf.placeholder(shape = inputs_tensor.get_shape(),
                                              dtype = inputs_tensor.dtype)
        self.dist = tf.reduce_sum(tf.square(self.closest - self.original_inputs))
        self.d_grad = tf.gradients(self.dist, self.closest)[0]
        
        # define session
        self.sess = sess

    def compute_margin(self, inputs, targets,
                       other_placeholder_values = [],
                       num_iterations = 200):
        ''' 
        compute the closest points to the boundaries between other classes

        Args:
        inputs - the input value, should be compatible with inputs_tensor
        targets - the target value of the network
        other_placeholder_values - a list of the values of other placeholders in the network
        num_iterations - number of iterations of gradient descent
    
        Returns:
        a list of distances of the class boundaries
        '''

        feed_dict = dict(zip([self.inputs_tensor, 
                              self.targets_tensor, 
                              self.original_inputs] + self.other_placeholders,
                             [inputs, targets, inputs] + other_placeholder_values))
        
        l = self.sess.run(self.logits,
                          feed_dict = feed_dict)
        pred_class = np.argmax(l)
        
        # define closest distance
        closest_dist = tf.reduce_sum(tf.square(self.closest - inputs))
        
        
        # traverse all the classes and compute the distance
        dist = [0] * self.num_classes
        closest_point = [0] * self.num_classes
        for c in xrange(self.num_classes):
            if c != pred_class:
                
                # set the vairable to the inputs
                self.sess.run(tf.assign(self.closest, inputs), feed_dict = feed_dict)
                
                # start gradient descent iterations
                for i in xrange(num_iterations):
                    # evaluate the current values of closest
                    closest_value = self.closest.eval(session = self.sess)
                    
                    # modify feed_dict
                    feed_dict[self.inputs_tensor] = closest_value
                    
                    # run some statistics
                    l_pred, l_c, cd = self.sess.run([self.logits[0, pred_class],
                                                     self.logits[0, c],
                                                     self.dist],
                                                    feed_dict = feed_dict)
                    l_diff = l_pred - l_c
                    print('Iteration {}: logit diff = {}, distance = {}'.format(i, l_diff, cd))
                    
                    # compute gradients
                    l_grad_pred, l_grad_c = self.sess.run([self.l_grad[pred_class], 
                                                           self.l_grad[c]], 
                                                          feed_dict = feed_dict)
                    # gradient 1 is the gradient away from the boundary
                    grad1 = 2 * l_diff * (l_grad_pred - l_grad_c)
                    
                    # gradient 2 is the gradient away from inputs
                    grad2 = self.sess.run(self.d_grad, feed_dict = feed_dict)
                    
                    # project gradient 2 onto the orthogonal space to grad1
                    proj = np.sum(grad1 * grad2) / \
                           np.sum(grad1 * grad1) * \
                           grad1
                    grad2 = grad2 - proj
                    
                    # combine the two gradients to the final gradient
                    grad = grad1 + grad2
                    
                    # run gradient descent
                    feed_dict[self.gradients] = grad
                    self.sess.run(self.apply_gradients, 
                                  feed_dict = feed_dict)
                
                
                ## final evaluation
                # evaluate the current values of closest
                closest_value = self.closest.eval(session = self.sess)

                # modify feed_dict
                feed_dict[self.inputs_tensor] = closest_value
                dist[c] = self.sess.run(closest_dist, feed_dict = feed_dict)
                closest_point[c] = closest_value
                
        return dist, closest_point, pred_class
                
            