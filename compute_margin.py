import tensorflow as tf
import numpy as np

class margin():
    def __init__(self, logits, inputs_tensor,
                 sess,
                 other_placeholders = [],
                 batch_size = 1):
        '''
        build the computation graph for distance computation
        
        Args:
        logits - the logit tensor in a network
        inputs_tensor - the input tensor in a network
        sess - tensorflow session
        other_placeholders - a list of other placeholders in the network
        batch_size - batch size
        '''
        
        # determine the number of classes
        self.num_classes = logits.get_shape().as_list()[-1]
        
        # define a trainable parameter
        self.batch_size = batch_size
        self.closest = tf.get_variable('closest', 
                                       shape = [self.batch_size]
                                       + inputs_tensor.get_shape().as_list()[1:])
        
        # placeholders
        print('Defining placeholders...')
        self.logits = logits
        self.inputs_tensor = inputs_tensor
        self.other_placeholders = other_placeholders
        
        # create optimizer & graident placeholder
        print('Creating optimizer and gradient placeholders...')
        with tf.variable_scope('margin_optimizer'):
            self.learning_rate = tf.constant(0.01)
            self.learning_rate1 = tf.constant(0.01)
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.optimizer1 = tf.train.AdamOptimizer(learning_rate = self.learning_rate1)
            self.gradients = tf.placeholder(shape = inputs_tensor.get_shape(),
                                            dtype = tf.float32)
            self.compute_gradients = [(self.gradients, self.closest)]
            self.apply_gradients = self.optimizer.apply_gradients(self.compute_gradients)
            self.apply_gradients1 = self.optimizer1.apply_gradients(self.compute_gradients)
        
        # define logistic gradients
        print('Defining logits gradients...')
        self.grad_class = tf.placeholder(shape = [None], dtype = tf.int32)
        self.selected_logit = tf.reduce_sum(self.logits
                                            * tf.one_hot(self.grad_class, 
                                                         self.num_classes,
                                                         axis = -1))
        self.l_grad = tf.gradients(self.selected_logit,
                                   self.inputs_tensor)[0]
            
        # define distance gradients
        print('Defining distance metric and its gradients...')
        self.original_inputs = tf.placeholder(shape = inputs_tensor.get_shape(),
                                              dtype = inputs_tensor.dtype)
        self.dist_per_token = tf.reduce_sum(tf.square(self.closest - self.original_inputs), 
                                            axis = range(1, self.closest.get_shape().ndims))
        self.dist = tf.reduce_sum(self.dist_per_token)
        self.d_grad = tf.gradients(self.dist, self.closest)[0]
        
        
        # define session
        self.sess = sess
        
        # initialize optimizer variables
        self.sess.run(tf.variables_initializer([var for var in tf.global_variables() if 'margin_optimizer' in var.name]))

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
                            label = None,
                            num_iterations = 200,
                            top = 1,
                            stage = 0,
                            initial_adv = None,
                            verbose = False,
                            offset = 0):
        ''' 
        compute the closest points to the boundaries between other classes

        Args:
        inputs - the input value, should be compatible with inputs_tensor
        other_placeholder_values - a list of the values of other placeholders in the network
        label - a numpy array of labels, the rank can be 1D, if single label, or 2D, if multiple labels
        num_iterations - number of iterations of gradient descent
        top - the number of top classes that are not considered as adversarial class
        stage - stage 0 decreases the predicted class until the boundary is hit
                stage 1 minimizes the distance along the boundary
        initial_adv - initial adversarial value which, if not empty, initializes the margin adversarial. Should be the same size as inputs
        verbose - if True, outputs the details of the process
        offset - the logit difference that is considered as the boundary
    
        Returns:
        a list of distances of the class boundaries
        '''

        # batch size
        batch_size = inputs.shape[0]
        assert batch_size == self.batch_size
        
        # reset optimizer
        self.sess.run(tf.variables_initializer([var for var in tf.global_variables() if 'margin_optimizer' in var.name]))
        
        # convert stage to a list of stages
        stage = np.array([stage] * self.batch_size)
        
        # define feed_dict
        feed_dict = dict(zip([self.inputs_tensor,
                              self.original_inputs] + self.other_placeholders,
                             [inputs, inputs] + other_placeholder_values))
        
        # find the top classes
        if label is not None:
            if len(label.shape) == 1:
                top_classes = label[:, np.newaxis] # convert to rank 2
                top = 1
            else:
                top_classes = label
                top = label.shape[1]
                
            l = self.sess.run(self.logits,
                              feed_dict = feed_dict)
            pred_class = np.argpartition(l, -top)[:, -top:]
            is_correct = np.sum((pred_class == top_classes).astype(float), -1)
            
        else:
            l = self.sess.run(self.logits,
                              feed_dict = feed_dict)
            top_classes = np.argpartition(l, -top)[:, -top:]
        
        # initialize the adversarial variable
        if initial_adv is not None:
            # set the vairable to the initial_adv
            self.sess.run(self.closest.assign(initial_adv), 
                          feed_dict = feed_dict)
        else:
            # set the vairable to the inputs
            self.sess.run(self.closest.assign(inputs), 
                          feed_dict = feed_dict)
            
        # start gradient descent iterations
        for i in xrange(num_iterations):
            # evaluate the current values of closest
            closest_value = self.closest.eval(session = self.sess)

            # evaluate the logit at largest point
            feed_dict[self.inputs_tensor] = closest_value
            l = self.sess.run(self.logits,
                             feed_dict = feed_dict)
            
            # locate the pred class
            l_tops = [l[j, top_classes[j, :]] 
                      for j in xrange(batch_size)]
            _top = [np.argmax(l_tops[j]) 
                    for j in xrange(batch_size)]
            pred_class = np.array([top_classes[j, _top[j]]
                                  for j in xrange(batch_size)])
            
            # locate the adversarial class
            l_argpart = np.argpartition(l, -2)[:, -top-1: ]
            l_argadvs = [np.setdiff1d(l_argpart[j, :], top_classes[j, :])
                         for j in xrange(batch_size)] # find the non top classes
            l_advs = [l[j, l_argadvs[j]]
                      for j in xrange(batch_size)]
            _adv = [np.argmax(l_advs[j])
                    for j in xrange(batch_size)]
            c = np.array([l_argadvs[j][_adv[j]]
                          for j in xrange(batch_size)])
            
            l_diff = [l[j, pred_class[j]] - l[j, c[j]]  - offset
                      for j in xrange(batch_size)]
            
            # change the stage if boundary is hit
            stage[list(stage == 0) and [ld < 0 for ld in l_diff]] = 1
            
            if verbose:
                # run some statistics
                cd = self.sess.run(self.dist_per_token,
                                   feed_dict = feed_dict)
                print('Iteration {}: on boundary against class {}, logit diff = {}, distance = {}'.format(i, c, l_diff, cd))

            # compute gradients
            feed_dict[self.grad_class] = pred_class
            l_grad_pred = self.sess.run(self.l_grad, 
                                        feed_dict = feed_dict)
            
            if np.any(stage == 1): # grad1 is useful only when there is stage 1
                feed_dict[self.grad_class] = c
                l_grad_c = self.sess.run(self.l_grad, 
                                         feed_dict = feed_dict)

                # gradient 1 is the gradient away from the boundary
                l_diff = np.reshape(np.array(l_diff), 
                                    tuple([-1] + [1] * (len(l_grad_pred.shape)-1)))
                grad1 = 2 * l_diff * (l_grad_pred - l_grad_c)
            else:
                grad1 = l_grad_pred

            if np.any(stage == 0) or i % 2 == 0:
                # stage 0: decrease the logit of pred class
                # stage 1: move the point towards the boundary
                grad_merge = grad1
                grad_merge[list(stage == 0), ...] = l_grad_pred[list(stage == 0), ...]
                feed_dict[self.gradients] = grad_merge
                self.sess.run(self.apply_gradients, 
                              feed_dict = feed_dict)
            
            else:
                # stage 0: no stage 0 in this direction
                # stage 1: move the point along the boundary towards the original input
                # gradient 2 is the gradient away from inputs
                grad2 = self.sess.run(self.d_grad, feed_dict = feed_dict)

                # project gradient 2 onto the orthogonal space to grad1
                # for stage 1 tokens only
                grad1_dir = l_grad_pred - l_grad_c

                proj = np.sum(grad1_dir * grad2) / \
                np.sum(grad1_dir * grad1_dir) * \
                grad1_dir
                grad2 = grad2 - proj
                feed_dict[self.gradients] = grad2
                self.sess.run(self.apply_gradients1, 
                              feed_dict = feed_dict)


        ## final evaluation
        # evaluate the current values of closest
        closest_value = self.closest.eval(session = self.sess)

        # produce final output
        feed_dict[self.inputs_tensor] = closest_value
        l_diff = np.reshape(l_diff, (-1))
        
        if label is not None: # introduce negative distance
            dist = (is_correct * 2 - 1) * self.sess.run(self.dist_per_token, 
                                                        feed_dict = feed_dict)
        else:
            dist = self.sess.run(self.dist_per_token, 
                                 feed_dict = feed_dict)
        closest_point = closest_value
        adv_class = c
        
                
        return dist, closest_point, pred_class, adv_class, l_diff
                
            
