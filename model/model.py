import os
from scipy.special import softmax

import sys
# import pyximport; pyximport.install()
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pickle
import time
import random
import numpy as np

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.core import Dense
from keras import backend as K
from keras.models import Model
from tensorflow.python.ops.gen_math_ops import mod
import keras
from keras.models import Model
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D, Dense, AveragePooling2D
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from skimage.transform import resize
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
folder_path = os.getcwd()


def  conv_model(num_labels, hidden_nodes=32, input_shape=(32,32,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same', use_bias=False, input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same',use_bias=False),
    keras.layers.Conv2D(32, (5,5), (2, 2),activation=tf.nn.relu,
                           padding='same',use_bias=False),
 

    keras.layers.Flatten(),
    keras.layers.Dense(200),
    keras.layers.ReLU(),
    keras.layers.Dense(100 ,activation = tf.nn.sigmoid, name='after_flatten'),
    # keras.layers.ReLU(max_value = 1,  name='after_flatten'),
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


def loihi_conv_model():
    
    ann_model = conv_model(9)
    with tf.device('/cpu:0'):
        ann_model.load_weights(folder_path +"/model/mnist_cnn.h5")
    
    intermediate_layer_model = Model(inputs=ann_model.input,
                                 outputs=ann_model.get_layer('after_flatten').output)

    return intermediate_layer_model

