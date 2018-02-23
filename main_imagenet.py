# load mudules
import tensorflow as tf
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib, cStringIO
import os

# load the model
model = tf.keras.applications.ResNet50()

# load compute_margin class
from compute_margin import margin
from tensorflow.python.keras._impl.keras import backend as K
m = margin(tf.log(model.output), model.input, K.get_session(), [K.learning_phase()])

# open the file
i = 0
#f = open('/mnt/nfs/nfsshare/user_homes/zybill/imagenet_data/fall11_urls.txt')
f = open('./imagenet_data/fall11_urls.txt')

# make the write directory
#write_dir = '/mnt/nfs/nfsshare/user_homes/zybill/results_imagenet/'
write_dir = './results/imagenet_adversarials/'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# run compute margin
input_shape = (224, 224)
for i in xrange(100):
    
    # load image
    input_shape = (224, 224)
    url = f.readline().split()[1]
    fl = cStringIO.StringIO(urllib.urlopen(url).read())
    image = tf.keras.preprocessing.image.load_img(fl, target_size = input_shape)
    image = tf.keras.preprocessing.image.img_to_array(image)
    im = np.array(image / 256)
    image = np.expand_dims(image, axis=0)

    # preprocess images
    image = tf.keras.applications.resnet50.preprocess_input(image)

    # classify the image to obtain the original class
    preds = model.predict(image)
    a = tf.keras.applications.resnet50.decode_predictions(preds, top=1)

    # plot the original image
    plt.imshow(im)
    plt.savefig(write_dir+'{}_or_{}.jpg'.format(i, a[0][0][1]))
    
    # compute the margins
    dist, closest_point, pred_class, adv_class, l_diff = m.compute_margin_fast(image, [False],
                                                                               num_iterations = 500,
                                                                               top = 50)

    # obtain the adversarial class
    preds = np.zeros((1, m.num_classes))
    preds[0, adv_class] = 1

    a = tf.keras.applications.resnet50.decode_predictions(preds, top=1)

    # anti-preprocessing
    cp = closest_point
    cp[..., 0] += 103.939
    cp[..., 1] += 116.779
    cp[..., 2] += 123.68
    cp = cp[0, ..., ::-1] / 256

    # plot image
    plt.imshow(cp)
    if l_diff < 0.01:
        plt.savefig(write_dir+'{}_ad_{}_dist{}.jpg'.format(i, a[0][0][1], dist))
    else:
        plt.savefig(write_dir+'{}_adfail_{}.jpg'.format(i, a[0][0][1]))
        
# close the file
f.close()