import model.model as md
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
from emstdp import loihi2_net

train_size = 20000
test_size =1000

# raw mnist data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:train_size]
y_train = y_train[:train_size]

#prepare pre-trained data from pretrained model
model = md.loihi_conv_model()
x_train = model(x_train)
x_train = x_train.numpy()

# prepare incremental learning tasks data
task_1 = [0,1]
task_2 = [2,3]
task_3 = [4,5]
task_4 = [6,7]
task_5 = [8,9]

task_1_data = []
task_2_data = []
task_3_data = []
task_4_data = []
task_5_data = []


task_1_label = []
task_2_label = []
task_3_label = []
task_4_label = []
task_5_label = []


full_data = 5*[None]
for i in range(len(x_train)):
    data =  np.expand_dims(x_train[i],axis =0)
    if y_train[i] in task_1:
        if len(task_1_data) ==0:
            task_1_data = data
        else:
            task_1_data = np.vstack((task_1_data,data))
        task_1_label.append(y_train[i])
    elif y_train[i] in task_2:
        if len(task_2_data) ==0:
            task_2_data = data
        else:
            task_2_data = np.vstack((task_2_data,data))
        task_2_label.append(y_train[i])

    elif y_train[i] in task_3:
        if len(task_3_data) ==0:
            task_3_data = data
        else:
            task_3_data = np.vstack((task_3_data,data))
        task_3_label.append(y_train[i])
    elif y_train[i] in task_4:
        if len(task_4_data) ==0:
            task_4_data = data
        else:
            task_4_data = np.vstack((task_4_data,data))
        task_4_label.append(y_train[i])
    else:
        if len(task_5_data) ==0:
            task_5_data = data
        else:
            task_5_data = np.vstack((task_5_data,data))
        task_5_label.append(y_train[i])

full_data = [[task_1_data, task_1_label],[task_2_data, task_2_label],[task_3_data, task_3_label],[task_4_data, task_4_label],[task_5_data, task_5_label]]

from emstdp import loihi2_net
net = loihi2_net([200,100,10],time_steps = 32)


test_set = [x_train[:10],y_train[:10]]
w_h, w_o=net.train_loihi_network(test_set)
del net

accuracy = []
epochs = 1
current_task_size = 400
memory_size = 400
memory_set = []
test_size = 1000
final_acc = []
w_h = []
w_o = []
for i in range(len(full_data)):
    current_set = full_data[i]
    for e in range(epochs):
        print("current epoch is ", e)
        for j in range(len(current_set)//current_task_size):
            #train current set
            net = loihi2_net([200,100,10],time_steps = 32,w_h = w_h,w_o=w_o)
            set_1 = [current_set[0][j*current_task_size:(j+1)*current_task_size],current_set[1][j*current_task_size:(j+1)*current_task_size]]
            w_h, w_o=net.train_loihi_network(set_1)
            del net
            net = loihi2_net(dim =[200,100,10],w_h =w_h, w_o= w_o,time_steps = 32)
            #train memory set
            if j%2==0:
                if len(memory_set)!=0:
                     print("memory set length", len(memory_set[0]))
                     w_h, w_o=net.train_loihi_network(memory_set)
                     del net
                print("done training iteration",j)
            
    #memory setttings
    if len(memory_set) ==0:
        x = current_set[0][:memory_size]
        y = current_set[1][:memory_size]
        memory_set = [x,y]
    else:
        x = np.vstack((memory_set[0],current_set[0][:memory_size]))
        y = np.append(memory_set[1],current_set[1])
        memory_set = [x,y]
        
#Testing 
net = loihi2_net(dim =[200,100,10],w_h =w_h, w_o= w_o,time_steps = 32)
test_set = [x_train[:test_size], y_train[:test_size]]
acc =net.test_loihi(test_set)
print("average acc of 5 tasks is ", acc)    




