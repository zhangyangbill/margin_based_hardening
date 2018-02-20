import tensorflow as tf
import numpy as np

class margin():
    def __init__(self, logits, inputs_tensor,
                 sess,
                 other_placeholders = []):
        '''
        build the computation graph for distance computation
        
        Args:
        logits - the logit tensor in a network
        inputs_tensor - the input tensor in a network
        sess - tensorflow session
        other_placeholders - a list of other placeholders in the network
        '''
        
        # determine the number of classes
        self.num_classes = logits.get_shape().as_list()[-1]
        
        # define a trainable parameter
        self.closest = tf.get_variable('closest', 
                                       shape = [1] + inputs_tensor.get_shape().as_list()[1:])
        
        # placeholders
        print('Defining placeholders...')
        self.logits = logits
        self.inputs_tensor = inputs_tensor
        self.other_placeholders = other_placeholders
        
        # create optimizer & graident placeholder
        print('Creating optimizer and gradient placeholders...')
        self.learning_rate = tf.constant(0.01)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate, name = 'optimizer')
        self.optimizer1 = tf.train.GradientDescentOptimizer(learning_rate = 0.01, name = 'optimizer')
        self.gradients = tf.placeholder(shape = inputs_tensor.get_shape(),
                                        dtype = tf.float32)
        self.compute_gradients = [(self.gradients, self.closest)]
        self.apply_gradients = self.optimizer.apply_gradients(self.compute_gradients)
        self.apply_gradients1 = self.optimizer1.apply_gradients(self.compute_gradients)
        
        # define logistic gradients
        print('Defining logits gradients...')
        self.grad_class = tf.placeholder(shape = [], dtype = tf.int32)
        self.selected_logit = tf.reduce_sum(self.logits[0, :] 
                                            * tf.one_hot(self.grad_class, 
                                                         self.num_classes), 
                                            axis = -1)
        self.l_grad = tf.gradients(self.selected_logit,
                                   self.inputs_tensor)[0]
            
        # define distance gradients
        print('Defining distance metric and its gradients...')
        self.original_inputs = tf.placeholder(shape = inputs_tensor.get_shape(),
                                              dtype = inputs_tensor.dtype)
        self.dist = tf.reduce_sum(tf.square(self.closest - self.original_inputs))
        self.d_grad = tf.gradients(self.dist, self.closest)[0]
        
        
        # define session
        self.sess = sess
        
        # initialize optimizer variables
        self.sess.run(tf.variables_initializer([var for var in tf.global_variables() if 'optimizer' in var.name]))

    def compute_margin(self, inputs,
                       other_placeholder_values = [],
                       num_iterations = 200):
        ''' 
        compute the closest points to the boundaries between other classes

        Args:
        inputs - the input value, should be compatible with inputs_tensor, only support batch size 1
        other_placeholder_values - a list of the values of other placeholders in the network
        num_iterations - number of iterations of gradient descent
    
        Returns:
        dist - a list of distances of the class boundaries
        closest_point - a list of closest point on the boundary to the input points
        pred_class - a scalar of predicted class
        '''

        feed_dict = dict(zip([self.inputs_tensor, 
                              self.original_inputs] + self.other_placeholders,
                             [inputs, inputs] + other_placeholder_values))
        
        l = self.sess.run(self.logits,
                          feed_dict = feed_dict)
        pred_class = np.argmax(l)
        
        
        # traverse all the classes and compute the distance
        dist = [0] * self.num_classes
        closest_point = [0] * self.num_classes
        for c in xrange(self.num_classes):
            if c != pred_class:
                
                # set the vairable to the inputs
                self.sess.run(self.closest.assign(inputs), 
                              feed_dict = feed_dict)
                
                # start gradient descent iterations
                for i in xrange(num_iterations):
                    # evaluate the current values of closest
                    closest_value = self.closest.eval(session = self.sess)
                    
                    # modify feed_dict
                    feed_dict[self.inputs_tensor] = closest_value
                    
                    # run some statistics
                    l, cd = self.sess.run([self.logits,
                                           self.dist],
                                           feed_dict = feed_dict)
                    l_diff = l[0, pred_class] - l[0, c]
                    print('Iteration {}: logit diff = {}, distance = {}'.format(i, l_diff, cd))
                    print(l[0, pred_class])
                    print(l[0, c])
                    
                    # compute gradients
                    feed_dict[self.grad_class] = pred_class
                    l_grad_pred = self.sess.run(self.l_grad, 
                                                feed_dict = feed_dict)
                    feed_dict[self.grad_class] = c
                    l_grad_c = self.sess.run(self.l_grad, 
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
                    
                    # run gradient descent
                    feed_dict[self.gradients] = grad1
                    self.sess.run(self.apply_gradients, 
                                  feed_dict = feed_dict)
                    feed_dict[self.gradients] = grad2
                    self.sess.run(self.apply_gradients1, 
                                  feed_dict = feed_dict)
                
                
                ## final evaluation
                # evaluate the current values of closest
                closest_value = self.closest.eval(session = self.sess)

                # modify feed_dict
                feed_dict[self.inputs_tensor] = closest_value
                dist[c] = self.sess.run(self.dist, feed_dict = feed_dict)
                closest_point[c] = closest_value
                
        return dist, closest_point, pred_class
    
    def compute_margin_fast(self, inputs,
                       other_placeholder_values = [],
                       num_iterations = 200):
        ''' 
        compute the closest points to the boundaries between other classes

        Args:
        inputs - the input value, should be compatible with inputs_tensor
        other_placeholder_values - a list of the values of other placeholders in the network
        num_iterations - number of iterations of gradient descent
    
        Returns:
        a list of distances of the class boundaries
        '''

        feed_dict = dict(zip([self.inputs_tensor,
                              self.original_inputs] + self.other_placeholders,
                             [inputs, inputs] + other_placeholder_values))
        
        l = self.sess.run(self.logits,
                          feed_dict = feed_dict)
        pred_class = np.argmax(l)
        second_largest_class = np.argpartition(l, -2)[0, -2]
        
        
        # set the vairable to the inputs
        self.sess.run(self.closest.assign(inputs), 
                      feed_dict = feed_dict)
        
        ## stage 1: decrease the class target until largest class is not pred_class
        #largest_class = pred_class
        #while largest_class == pred_class:
        #    # evaluate the gradient towards pred_class
        #    feed_dict[self.grad_class] = pred_class
        #    l_grad_pred = self.sess.run(self.l_grad, 
        #                                feed_dict = feed_dict)
        #    feed_dict[self.grad_class] = second_largest_class
        #    l_grad_2largest = self.sess.run(self.l_grad, 
        #                                    feed_dict = feed_dict)
        #    
        #    # run gradient descent
        #    feed_dict[self.gradients] = l_grad_pred - l_grad_2largest
        #    self.sess.run(self.apply_gradients, 
        #                  feed_dict = feed_dict)
        #    
        #    # evaluate the current values of closest
        #    closest_value = self.closest.eval(session = self.sess)
#
        #    # modify feed_dict
        #    feed_dict[self.inputs_tensor] = closest_value
        #    
        #    # reevaluate the logits to see if pred_class is still the largest one
        #    l = self.sess.run(self.logits,
        #                      feed_dict = feed_dict)
        #    largest_class = np.argmax(l)
        #    
        #    # print some statistics:
        #    second_largest_class = np.argpartition(l, -2)[0, -2]
        #    print('Largest class: {}, second largest class: {}, logit difference = {}'.format(largest_class, 
        #                                                                                      second_largest_class,
        #                                                                                      l[0, largest_class] - l[0, second_largest_class]))
            
        # stage 2: move along the class boundaries towards the inputs
        # start gradient descent iterations
        for i in xrange(num_iterations):
            # evaluate the current values of closest
            closest_value = self.closest.eval(session = self.sess)

            # modify feed_dict
            feed_dict[self.inputs_tensor] = closest_value

            # locate the largest two classes of logits
            l = self.sess.run(self.logits,
                             feed_dict = feed_dict)
            l_part = np.argpartition(l, -2)
            # locate the largest non-pred_class class
            if l_part[0, -1] == pred_class:
                c = l_part[0, -2]
            else:
                c = l_part[0, -1]
            
            # run some statistics
            l_diff = l[0, pred_class] - l[0, c]
            cd = self.sess.run(self.dist,
                               feed_dict = feed_dict)
            print('Iteration {}: on boundary against class {}, logit diff = {}, distance = {}'.format(i, c, l_diff, cd))

            # compute gradients
            feed_dict[self.grad_class] = pred_class
            l_grad_pred = self.sess.run(self.l_grad, 
                                        feed_dict = feed_dict)
            feed_dict[self.grad_class] = c
            l_grad_c = self.sess.run(self.l_grad, 
                                     feed_dict = feed_dict)
            # gradient 1 is the gradient away from the boundary
            grad1 = 2 * l_diff * (l_grad_pred - l_grad_c)
            
            # compute the norm of gradient 1 to adjust learning rate
            if i == 0:
                grad1_norm = np.sqrt(np.sum(grad1 ** 2))
                feed_dict[self.learning_rate] = 1 / grad1_norm

            # gradient 2 is the gradient away from inputs
            grad2 = self.sess.run(self.d_grad, feed_dict = feed_dict)

            # project gradient 2 onto the orthogonal space to grad1
            proj = np.sum(grad1 * grad2) / \
            np.sum(grad1 * grad1) * \
            grad1
            grad2 = grad2 - proj

            # run gradient descent
            feed_dict[self.gradients] = grad1
            self.sess.run(self.apply_gradients, 
                          feed_dict = feed_dict)
            feed_dict[self.gradients] = grad2
            self.sess.run(self.apply_gradients1, 
                          feed_dict = feed_dict)


            ## final evaluation
            # evaluate the current values of closest
            closest_value = self.closest.eval(session = self.sess)

            # modify feed_dict
            feed_dict[self.inputs_tensor] = closest_value
            dist = self.sess.run(self.dist, feed_dict = feed_dict)
            closest_point = closest_value
            
            adv_class = c
                
        return dist, closest_point, pred_class, adv_class
                
            
