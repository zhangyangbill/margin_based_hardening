import tensorflow as tf
import numpy as np
from compute_margin import margin
import math
import matplotlib.pyplot as plt #### test!!!!####

class margin_based_defense:
    
    def __init__(self, 
                 logits, 
                 inputs_tensor,
                 targets_tensor,
                 loss,
                 sess,
                 other_placeholders = [],
                 update_percentage = 0.3,
                 adv_batch_size = 1,
                 learning_rate = 1e-3,
                 one_hot = True):
        '''
        build the computation graph for distance computation
        
        Args:
        logits - the logit tensor in a network
        inputs_tensor - the input tensor in a network
        targets_tensor - the target tensor in a network
        loss - the loss tensor in the network
        sess - tensorflow session
        other_placeholders - a list of other placeholders in the network
        update_percentage - the percentage of the training data whose margin get maximized
        adv_batch_size - the batch size for computing adversarial samples
        learning_rate - the learning rate of the optimizer
        one_hot - if the targets is one_hot encoded
        '''
        
        # define class variables
        self.logits = logits
        self.inputs = inputs_tensor
        self.targets = targets_tensor
        self.loss = loss
        self.sess = sess
        self.other_placeholders = other_placeholders
        self.update_percentage = update_percentage
        self.adv_batch_size = adv_batch_size
        self.one_hot = one_hot
        self.num_classes = self.logits.get_shape().as_list()[-1]
        
        # initialize a margin class
        self.margin = margin(self.logits,
                             self.inputs,
                             self.sess,
                             other_placeholders = self.other_placeholders,
                             batch_size = adv_batch_size)
        
        # record margin information
        self.small_margins = []
        self.small_margin_ids = []
        self.small_position = []  # position of each token in the small_margins list
        self.recent_small_margins = []
        self.recent_small_margins_head = 0
        self.max_margin = -np.inf
        self.max_margin_id = 0
        self.num_small_margins = 0
        self.max_num_small_margins = 0
        
        
        # training operation
        with tf.variable_scope('defense_optimizer'):
            self.optim = tf.train.AdamOptimizer()
            self.train_op = self.optim.minimize(loss)
            self.grads_vars = self.optim.compute_gradients(loss)
            self.grads = [gv[0] for gv in self.grads_vars if gv[0] is not None]
            self.apply_gradients = self.optim.apply_gradients(self.grads_vars)
            
        self.pred = tf.argmax(logits, 1)
        if self.one_hot:
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,
                                                            tf.argmax(self.targets, 1)),
                                                   tf.float32))
        else:
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,
                                                            self.targets),
                                                   tf.float32))
        
        # initialize optimizer variables
        self.sess.run(tf.variables_initializer([var for var in tf.global_variables() if 'defense_optimizer' in var.name]))
        
        
    def _proj_gradients(self,
                        inputs,
                        targets,
                        other_placeholder_values = [],
                        max_num_bases = 3):
        '''
        Project the gradients onto the orthogonal subspace of the individual gradients whose inner product is negative
        
        Args:
        inputs - the input features for classification
        targets - the targets
        other_placeholder_values - a list of other placeholders in the network
        max_num_bases - max number of basis onto which the gradients project, should be small
        '''
        
        
        # run individual gradients
        train_batch_size = inputs.shape[0]
        grads_indv = []
        for i in xrange(train_batch_size):
            feed_dict = dict(zip([self.inputs, self.targets]
                                 + self.other_placeholders,
                                 [inputs[i:i+1, ...],
                                  targets[i:i+1, ...]]
                                 + other_placeholder_values))
            grads_indv.append(self.sess.run(self.grads, feed_dict = feed_dict))
            
        # compute aggregate gradients
        grads_agg = [sum([g[j] for g in grads_indv]) / train_batch_size
                     for j in xrange(len(grads_indv[0]))]
        
        # compute the inner product between aggregate gradients and each individual gradients
        inner_prods = np.array([sum([np.sum(ga * gi) for ga, gi in zip(grads_agg, grads_i)])
                                for grads_i in grads_indv])
        
        # pick the negative inner prods, up to max_num_bases
        neg_id = np.argpartition(inner_prods, max_num_bases)[0:max_num_bases]
        _neg = inner_prods[neg_id]
        _is_neg = list(_neg < 0)
        neg_id = neg_id[_is_neg]
        neg_inner_prods = _neg[_is_neg]
        neg_grads_indv = [grads_indv[i] for i in list(neg_id)]
        
        # perform projection
        if neg_id.shape[0] != 0:
            
            # compute autocorrelation matrix
            # vectorize each gradient matrix
            _vec_gradients = [[np.reshape(gg, (1, -1)) for gg in g]
                              for g in neg_grads_indv]
            # concatenate the gradient vectors of different tokens
            _vec_gradients = [np.concatenate(g, 0) for g in zip(*_vec_gradients)]
            
            autocorr = sum([np.matmul(g, g.T) for g in _vec_gradients])
            
            try:
                # compute projection coefficients
                proj_coef = list(np.matmul(np.linalg.inv(autocorr), neg_inner_prods[:, np.newaxis])[:, 0])

                # compute projected gradients
                grads_proj = [g_bar - sum([g[j] * p for g, p in zip(neg_grads_indv, proj_coef)])
                          for g_bar, j in zip(grads_agg, xrange(len(grads_agg)))]
            
            except: # if sigular matrix error is encountered
                grads_proj = grads_agg
            
        else:
            grads_proj = grads_agg
            
        # apply the projected graidents
        feed_dict = dict(zip([self.inputs, self.targets]
                             + self.other_placeholders
                             + self.grads,
                             [inputs, targets]
                             + other_placeholder_values
                             + grads_proj))
        self.sess.run(self.apply_gradients, feed_dict = feed_dict)
        
           
    
    def _update_margin(self,
                       inputs,
                       inputs_adv,
                       inputs_id,
                       targets,
                       other_placeholder_values = [],
                       top = 1,
                       recompute_adv = False,
                       verbose = False,
                       offset = 0):
        '''
        trains a batch of input data by maximizing the margin
        
        Args:
        inputs - a numpy array of input features
        inputs_adv - a numpy array of the adversarial sample of each input
        inputs_id - a list of the input indentity of each token
        targets - a numpy array of labels
        other_placeholder_values - a list of other placeholders in the network
        top - the number of top classes that are not considered as adversarial class
        recompute_adv - If True, recompute the adversarial samples from scratch
        verbose - if True, outputs the details of the process
        offset - the logit difference that is considered as the boundary
        
        Returns:
        inputs_adv - an updated inputs adversarial samples
        '''
        
        batch_size = inputs.shape[0]
        
        # initialize feed_dict
        feed_dict = dict(zip([self.inputs] + self.other_placeholders,
                             [inputs] + other_placeholder_values))
        
        # update the adversarial samples
        
        if recompute_adv:
            # compute adversarial samples from scratch
            dists, inputs_adv, _,_,_ = self.margin.compute_margin_fast(inputs,
                                                                       other_placeholder_values = other_placeholder_values,
                                                                       label = targets,
                                                                       num_iterations = 200,
                                                                       top = top,
                                                                       verbose = verbose,
                                                                       offset = offset)
        else:
            # incrementally update adversarial sample
            dists, inputs_adv, _,_,_ = self.margin.compute_margin_fast(inputs,
                                                                       other_placeholder_values = other_placeholder_values,
                                                                       label = targets,
                                                                       top = top,
                                                                       num_iterations = 200,
                                                                       #stage = 1,
                                                                       initial_adv = inputs_adv,
                                                                       verbose = verbose,
                                                                       offset = offset)
                    
        # update the small margin lists for all the tokens
        self._update_margin_lists(dists, inputs_id)
        
        # inputs_adv equal inputs if wrongly classified
        inputs_adv[list(dists < 0), ...] = np.array(inputs[list(dists < 0), ...])
        
        return inputs_adv
    
    def _update_margin_lists(self, dists, token_ids):
        ''' 
        Update self.small_margins, self.small_margins_id and self.is_small with one new token information
        
        Args:
        dists - a list of the distance of each token to the boundary
        token_ids - a list each token id
        '''
        
        for (dist, token_id) in zip(dists, token_ids):
            if self.small_position[token_id] is not None: # if already in the list
                self.small_margins[self.small_position[token_id]] = dist
                self.recent_small_margins[self.recent_small_margins_head] = token_id
                self.recent_small_margins_head = (self.recent_small_margins_head + 1) \
                                                 % self.train_batch_size
                
                # update the max margin list
                if dist > self.max_margin:
                    self.max_margin = dist
                    self.max_margin_id = token_id
            
            elif self.num_small_margins < self.max_num_small_margins: # if the small margin list is not full
                self.small_margins[self.num_small_margins] = dist
                self.small_margin_ids[self.num_small_margins] = token_id
                self.small_position[token_id] = self.num_small_margins
                self.num_small_margins += 1
                self.recent_small_margins[self.recent_small_margins_head] = token_id
                self.recent_small_margins_head = (self.recent_small_margins_head + 1) \
                                                 % self.train_batch_size
                
                # update the max margin list
                if dist > self.max_margin:
                    self.max_margin = dist
                    self.max_margin_id = token_id
        
            elif dist < self.max_margin: # is full but this one is better
                self.small_margins[self.small_position[self.max_margin_id]] = dist
                self.small_margin_ids[self.small_position[self.max_margin_id]] = token_id
                self.small_position[token_id] = self.small_position[self.max_margin_id]
                self.small_position[self.max_margin_id] = None # kick the original out
                self.recent_small_margins[self.recent_small_margins_head] = token_id
                self.recent_small_margins_head = (self.recent_small_margins_head + 1) \
                                                 % self.train_batch_size
            
                # update the max margin list
                _max_id = np.argmax(self.small_margins)
                self.max_margin = self.small_margins[_max_id]
                self.max_margin_id = self.small_margin_ids[_max_id]
                
        
    def train(self,
              inputs, targets, 
              other_placeholder_values = [],
              train_batch_size = 1,
              num_epochs = 1,
              test_inputs = None,
              test_targets = None):
        
        ''' Perform margin-based training
        
        Args:
        inputs - a numpy array of input features
        targets - a numpy array of targets (two-hot)
        other_placeholder_values - a list of other placeholders in the network
        train_batch_size - train batch size
        num_epochs - number of training epochs
        test_inputs - a numpy array of input features from the test data
        test_targets - a numpy array of targets from the test data
        
        '''
        
        # make sure adv_batch_size >= train_batch_size
        self.train_batch_size = train_batch_size
        assert self.adv_batch_size >= train_batch_size
        
        # number of training tokens
        num_tokens = inputs.shape[0]
        
        # convert one-hot target into non-one-hot
        if self.one_hot:
            targets_noh = np.argmax(targets, axis = 1)
        else:
            targets_noh = targets
            
        
        # initialize small margin lists
        self.max_num_small_margins = int(round(self.update_percentage * num_tokens))
        self.small_margins = [None] * self.max_num_small_margins
        self.small_margin_ids = [None] * self.max_num_small_margins
        self.small_position = [None] * num_tokens
        self.recent_small_margins = [None] * self.train_batch_size
        
        # initialize input adversarials
        inputs_adv = np.array(inputs)
        
        # number of batches
        num_batches_adv = int(math.floor(float(num_tokens) / self.adv_batch_size))
        
        for epoch_id in xrange(num_epochs):
            
            for adv_id in xrange(num_batches_adv):
                
                # shuffle tokens
                shuffle_adv = np.random.permutation(num_tokens)

                if epoch_id % 10 == 0:
                    recompute = True
                else:
                    recompute = False
                
                # find adversarials
                adv_indices = shuffle_adv[adv_id * self.adv_batch_size : (adv_id+1) * self.adv_batch_size]
                

                inputs_adv[adv_indices, ...] = self._update_margin(inputs[adv_indices, ...],
                                                                   inputs_adv[adv_indices, ...],
                                                                   adv_indices,
                                                                   targets_noh[adv_indices],
                                                                   other_placeholder_values = other_placeholder_values,
                                                                   recompute_adv = recompute,
                                                                   verbose = False)

                ################# test!!!! #########
                # define the grid
                _grid_x = np.arange(-4, 4, 0.05)
                _grid_y = np.arange(-4, 4, 0.05)

                grid_x, grid_y = np.meshgrid(_grid_x, _grid_y)
                grid = np.concatenate((grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1))), axis = -1)

                pred_grid = self.sess.run(self.pred, 
                                          feed_dict = {self.inputs: grid}).reshape((_grid_x.shape[0], 
                                                                                    _grid_x.shape[0]))
                pred_boundary = np.logical_or((np.diff(pred_grid, axis = 0) != 0)[:, 0:-1],
                                              (np.diff(pred_grid, axis = 1) != 0)[0:-1, :])

                plt.imshow(pred_boundary, extent = [-4, 4, 4, -4], cmap = 'Greys')
                plt.hold(True)
                plt.scatter(inputs[list(targets == 0), 0], inputs[list(targets == 0), 1], marker = 'x', c = 'b')
                plt.scatter(inputs[list(targets == 1), 0], inputs[list(targets == 1), 1], marker = 'o', c = 'b')
                adv0_id = [rr for rr in self.recent_small_margins if targets[rr] == 0]
                adv1_id = [rr for rr in self.recent_small_margins if targets[rr] == 1]
                plt.scatter(inputs_adv[adv0_id, 0], 
                            inputs_adv[adv0_id, 1],
                            marker = 'x', c = 'g')
                plt.scatter(inputs_adv[adv1_id, 0], 
                            inputs_adv[adv1_id, 1],
                            marker = 'o', c = 'g')
                plt.scatter(inputs[adv0_id, 0], 
                            inputs[adv0_id, 1],
                            marker = 'x', c = 'r')
                plt.scatter(inputs[adv1_id, 0], 
                            inputs[adv1_id, 1],
                            marker = 'o', c = 'r')
                
                plt.savefig('./results/scatter_plots2/epoch{}_batch{}_before.png'.format(epoch_id, adv_id))
                plt.close()
                ############### test!!!!!! ###############
                
                
                
                # perform training
                feed_dict = dict(zip([self.inputs, self.targets]
                                     + self.other_placeholders,
                                     [inputs_adv[self.recent_small_margins, ...],
                                      targets[self.recent_small_margins, ...]]
                                     + other_placeholder_values))
                
                #if epoch_id > 0:
                #    self.sess.run(self.train_op, feed_dict = feed_dict)
                if epoch_id > 0:
                    self._proj_gradients(inputs_adv[self.recent_small_margins, ...],
                                         targets[self.recent_small_margins, ...],
                                         other_placeholder_values)
                
                ################# test!!!! #########
                # define the grid
                _grid_x = np.arange(-4, 4, 0.05)
                _grid_y = np.arange(-4, 4, 0.05)

                grid_x, grid_y = np.meshgrid(_grid_x, _grid_y)
                grid = np.concatenate((grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1))), axis = -1)

                pred_grid = self.sess.run(self.pred, 
                                          feed_dict = {self.inputs: grid}).reshape((_grid_x.shape[0], 
                                                                                    _grid_x.shape[0]))
                pred_boundary = np.logical_or((np.diff(pred_grid, axis = 0) != 0)[:, 0:-1],
                                              (np.diff(pred_grid, axis = 1) != 0)[0:-1, :])

                plt.imshow(pred_boundary, extent = [-4, 4, 4, -4], cmap = 'Greys')
                plt.hold(True)
                plt.scatter(inputs[list(targets == 0), 0], inputs[list(targets == 0), 1], marker = 'x', c = 'b')
                plt.scatter(inputs[list(targets == 1), 0], inputs[list(targets == 1), 1], marker = 'o', c = 'b')
                plt.scatter(inputs_adv[adv0_id, 0], 
                            inputs_adv[adv0_id, 1],
                            marker = 'x', c = 'g')
                plt.scatter(inputs_adv[adv1_id, 0], 
                            inputs_adv[adv1_id, 1],
                            marker = 'o', c = 'g')
                plt.scatter(inputs[adv0_id, 0], 
                            inputs[adv0_id, 1],
                            marker = 'x', c = 'r')
                plt.scatter(inputs[adv1_id, 0], 
                            inputs[adv1_id, 1],
                            marker = 'o', c = 'r')
                
                plt.savefig('./results/scatter_plots2/epoch{}_batch{}_after.png'.format(epoch_id, adv_id))
                plt.close()
                ############### test!!!!!! ###############
                

                # output information
                loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict = feed_dict)
                if self.num_small_margins == 0:
                    adv_dist_mean = 0
                else:
                    adv_dist_mean = sum(self.small_margins[0 : self.num_small_margins]) / self.num_small_margins
                print 'Epoch {}, batch {}: training loss = {}, training accuracy = {}, adverage adversarial distance = {}'.format(epoch_id, adv_id, loss, acc, adv_dist_mean)

                if test_inputs is not None and test_targets is not None:
                    feed_dict[self.inputs] = test_inputs
                    feed_dict[self.targets] = test_targets
                    loss_test, acc_test = self.sess.run([self.loss, self.accuracy], feed_dict = feed_dict)
                    print 'test loss = {}, test accuracy = {}'.format(loss_test, acc_test)
      