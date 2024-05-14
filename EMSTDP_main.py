import os

cpu_num = "2"       # choose number of cores to run the model (CPU)
os.environ["MKL_NUM_THREADS"] = cpu_num
os.environ["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS="+cpu_num
os.environ["OMP_NUM_THREADS"] = cpu_num
os.environ["MKL_DYNAMIC"] = "False"
os.environ["OMP_DYNAMIC"] = "False"

import sys
# import pyximport; pyximport.install()

import numpy as np
import EMSTDP_algo as svp
# import BPSNN_allSpikeErr_batcheventstdp2 as svp
import matplotlib.pyplot as plt
import pickle
import gzip
import time
import random

from tqdm import tqdm, tqdm_gui, tnrange, tgrange, trange
# import progressbar


## Import MNIST dataset (to load other dataset, format it to the MNIST format in Keras)

from keras.datasets import mnist                        # load MNIST dataset
# from keras.datasets import fashion_mnist as mnist     # load Fashion MNIST dataset

data_train = np.load("x_train.npy")
data_label = np.load("y_train.npy")

# _, h, w = np.shape(TRAINING[0])

total_train_size = 2000     # total number of training samples
total_test_size = 100   # total number of testing samples

train_size = 2000                       # verification period
test_size = 100                         # verification test size
ver_period = train_size

epochs = 10                             # number of epochs

dim = 200

## extract images and labels
data = np.expand_dims(np.reshape(data_train[0:total_train_size], [total_train_size,dim ]), axis=0)
label = np.argmax(data_label, axis=1).astype(int)

dataTest = np.expand_dims(np.reshape(data_train[0:total_test_size], [total_test_size, dim]), axis=0)
labelTest = np.zeros((total_test_size, 10))
labelTest = np.argmax(data_label[:total_test_size], axis=1).astype(int)

data_index = (np.linspace(0, total_train_size - 1, total_train_size)).astype(int)

print(data.shape)
# initialize hyper-parameters (the descriptions are in the Network class)
h = [100]  # [100,300,500,700,test_size,1500]

ind = -1

T = 64
twin = int( T / 2 - 1)
epsilon = 1
scale = 1.0
bias = 0.0
batch_size = 100
tbs = 100
#fr = 0.5
fr= 1
rel = 0
delt = 5
clp = True
lim = 1.0
dropr = 0.0

final_energy = np.zeros([epochs])

hiddenThr1 = 0.5
outputThr1 = 0.1

energies = np.zeros([train_size])
batch_energy = np.zeros([int(train_size / 50)])  # bach_size = 50
ind += 1
acc = []



lr = 0.003
# def __init__(self, dfa, dropr, evt, norm, rel, delt, dr, init, clp, lim, inputs, hiddens, outputs, threshold_h, threshold_o, T=100, bias=0.0, lr=0.0001, scale=1.0, twin=100, epsilon=2):
snn_network = svp.Network(0, dropr, 0, 0.0, rel, delt, 1, 0, clp, lim, 200, h, 10, hiddenThr1*fr, outputThr1*fr, T, bias, lr, scale, twin, epsilon)

snn_network.w_h = [np.load("w_h.npy")]
snn_network.w_o = np.load("w_o.npy")
fr = 1
s_index = data_index
print(s_index.shape)
# for ep in trange(epochs):
#     snn_network.lr = 0.003
#     pred1 = np.zeros([train_size])
#     outs = np.zeros([train_size,10])
#     # np.random.shuffle(s_index)
#     spikes = np.zeros([T, batch_size, dim]).astype(float)
#     spikes2 = np.zeros([T, batch_size, dim]).astype(float)
#     # for i in trange(train_size / batch_size, leave=False):
#     for i in trange(int (train_size / batch_size)):
#         # if ((i + 1)%5 == 0):  # 5000
#         #     pred = np.zeros([test_size])
#         #     for i2 in range(int(test_size / tbs)):  # train_size

#         #         tmp_rand = np.random.random([twin, 1, 1])
#         #         randy = np.tile(tmp_rand, (1, tbs,200))

#         #         tmp_d = np.tile(dataTest[:, i2 * tbs:(i2 + 1) * tbs, :], (twin, 1, 1))
#         #         spikes2 = randy < (tmp_d * fr)

#         #         pred[i2 * tbs:(i2 + 1) * tbs] = snn_network.Test(spikes2.astype(float), tbs)
#         #         print(labelTest[i2 * tbs:(i2 + 1) * tbs])
#         #     acn = sum(pred == labelTest[:test_size]) / float(test_size)
#         #     print( str(ep) + " test_accuray " + str(acn) + " LR " + str(snn_network.lr))
#         #     acc.append(sum(pred == labelTest[:test_size]) / float(test_size))


#         tmp_rand = np.random.random([T, 1, 1])
#         randy = np.tile(tmp_rand, (1, batch_size, dim))
#         tmp_d = np.tile(data[:, s_index[i * batch_size:(i + 1) * batch_size], :], (T, 1, 1))
#         spikes = randy < (tmp_d * fr)
#         pred1[i * batch_size:(i + 1) * batch_size], energies[i] = snn_network.Train(spikes.astype(float), (
#             label[s_index[i * batch_size:(i + 1) * batch_size]]), batch_size)
#         # sys.stdout.flush()
#     acn = sum(pred1 == label[s_index[:train_size]]) / float(train_size)
#     print(str(ep) + " train_accuray " + str(acn))
# np.save("w_h.npy", snn_network.w_h)
# np.save("w_o.npy",snn_network.w_o)


# print(snn_network.threshold_h)
# print(snn_network.threshold_o)

# print(snn_network.ethreshold_h) 
# print(snn_network.ethreshold_o)




# fr = 0.5




# tmp_rand = np.random.random([T, 1, 1])
# randy = np.tile(tmp_rand, (1, tbs,200))
# spikes2 = np.zeros([T, batch_size, dim]).astype(float)
# tmp_d = np.tile(dataTest[:, :tbs, :], (T, 1, 1))
# spikes2 = randy < (tmp_d * fr)
# print(spikes2.shape)
# res, train_new = snn_network.Test(spikes2.astype(float), tbs)
# print(train_new[1])
# np.save("train_new",train_new)
# print(res[:20])
# print(label[:20])
# np.save("label_new.npy",label[:100])