# load mudules
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# load the model
from mnist import mnist
model = mnist()
model.saver.restore(model.sess, "./tmp/model1.ckpt")

# load compute_margin class
from compute_margin import margin
m = margin(model.logits, model.x, model.y_, model.sess, [model.keep_prob])

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# run compute margin
for i in xrange(4, 100):
    batch = mnist.train.next_batch(1)
    dist, closest_point, pred_class = m.compute_margin(batch[0], batch[1], [1],
                                                       num_iterations = 200)
    
    # save data
    if not os.path.exists('/mnt/nfs/nfsshare/user_homes/zybill/results/{}/'.format(i)):
        os.makedirs('/mnt/nfs/nfsshare/user_homes/zybill/results/{}/'.format(i))
    with open('/mnt/nfs/nfsshare/user_homes/zybill/results/{}/dist.txt'.format(i), 'w+') as f:
        f.write('True class: {}, predicted class: {}\n'.format(np.argwhere(batch[1])[0, 1], pred_class))
        for c in xrange(10):
            f.write('Distance to class {}: {}\n'.format(c, dist[c]))
            
    plt.imshow(np.reshape(batch[0], (28, 28)), cmap = 'Greys')
    plt.savefig('/mnt/nfs/nfsshare/user_homes/zybill/results/{}/class_{}_original.png'.format(i, pred_class))
    for c in xrange(10):
        if c != pred_class:
            plt.imshow(np.reshape(closest_point[c], (28, 28)), cmap = 'Greys')
            plt.savefig('/mnt/nfs/nfsshare/user_homes/zybill/results/{}/class_{}.png'.format(i, c))
