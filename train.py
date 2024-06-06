import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.models import Model
import os
from scipy.ndimage.interpolation import rotate
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

 #the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print(y_train.shape)

x_train = np.load("imgs.npy")[:50000]
y_train = np.load("label.npy")[:50000]
np.save("original_x.npy",x_train)
print(x_train.shape)
print(y_train.shape)

print(np.max(x_train))
# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)



x_train = x_train.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_train = x_train/np.max(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

import keras
from keras.models import Model
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D, Dense, AveragePooling2D
def   simple_softmax_conv_model(num_labels, hidden_nodes=32, input_shape=(32,32,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same', use_bias=False, input_shape=input_shape),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='same',use_bias=False),
    keras.layers.Conv2D(hidden_nodes, (5,5), (2, 2),activation=tf.nn.relu,
                           padding='same',use_bias=False),
 
    keras.layers.Flatten(),
   
    keras.layers.Dense(200),
     keras.layers.ReLU(max_value = 1,  name='after_flatten'), 
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])


ann_model = simple_softmax_conv_model(num_labels=10)

# ann_model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# # ann_model.summary()

# # Training 
# ann_model.fit(x_train, y_train, 128, 20, verbose=2,
#         validation_data=(x_train, y_train))


# ann_model.save_weights("loihi_cnn.h5")
ann_model.load_weights("loihi_cnn.h5")
res = ann_model(x_train[:100])