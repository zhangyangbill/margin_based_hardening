# load mudules
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# load the model
model = tf.keras.applications.ResNet50()

# load compute_margin class
from compute_margin import margin
from tensorflow.python.keras._impl.keras import backend as K
m = margin(model.output, model.input, [], K.get_session(), [K.learning_phase()])

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# run compute margin
input_shape = (224, 224)
for i in xrange(1):
    # load image
    image = tf.keras.preprocessing.image.load_img('./Bend swage bead.jpg', target_size = input_shape)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # preprocess images
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    # compute the margins
    dist, closest_point, pred_class = m.compute_margin(image, [], [False],
                                                       num_iterations = 200)
    closest_class = np.argpartition(dist, 2)[1]
    
    # save data
    if not os.path.exists('/mnt/nfs/nfsshare/user_homes/zybill/results_imagenet/{}/'.format(i)):
        os.makedirs('/mnt/nfs/nfsshare/user_homes/zybill/results_imagenet/{}/'.format(i))
    with open('/mnt/nfs/nfsshare/user_homes/zybill/results_imagenet/{}/dist.txt'.format(i), 'w+') as f:
        f.write('True class: {}, predicted class: {}, closest class: {}\n'.format([], pred_class, closest_class)
        for c in xrange(len(dist)):
            f.write('Distance to class {}: {}\n'.format(c, dist[c]))
    
    # plot the original and the closest adversarial image        
    plt.imshow(np.reshape(batch[0], (28, 28)), cmap = 'Greys')
    plt.savefig('/mnt/nfs/nfsshare/user_homes/zybill/results_imagenet/{}/class_{}_original.png'.format(i, pred_class))
    c = closest_class
    plt.imshow(np.reshape(closest_point[c], (28, 28)), cmap = 'Greys')
    plt.savefig('/mnt/nfs/nfsshare/user_homes/zybill/results_imagenet/{}/class_{}.png'.format(i, c))
