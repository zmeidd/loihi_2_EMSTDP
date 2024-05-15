import numpy as np
import opt_einsum as oe
# process definition
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU, NeuroCore
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, tag, requires
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
'''
recieve post spikes 
if the IO works well on board
'''
def simple_softmax_conv_model(num_labels,  input_shape=(32,32,1), l2_reg=0.0):
    return keras.models.Sequential([
    keras.layers.Conv2D(16, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='valid', use_bias=False, input_shape=input_shape),
    keras.layers.Conv2D(8, (5,5), (2, 2), activation=tf.nn.relu,
                           padding='valid',use_bias=False),
 
    keras.layers.Flatten(),
    keras.layers.Dense(200),
    keras.layers.ReLU(max_value = 1,  name='after_flatten'), 
    keras.layers.Dense(100),
    keras.layers.ReLU(max_value = 1), 
    keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')])

def ann_model():
    
    ann_model = simple_softmax_conv_model(10)
    
    ann_model.load_weights("ann_wgt.h5")
    
    intermediate_layer_model = Model(inputs=ann_model.input,
                                 outputs=ann_model.get_layer('after_flatten').output)

    return intermediate_layer_model


class VecRecvProcess(AbstractProcess):
    """
    Process that receives arbitrary vectors

    Parameters
    ----------
    shape: tuple, shape of the process
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        self.shape = shape
        self.s_in = InPort(shape=(shape[1],))
        self.spk_data = Var(shape=shape, init=0)  # This Var expands with time
        
# @implements(proc=VecRecvProcess, protocol=LoihiProtocol)
# @requires(CPU)
# # need the following tag to discover the ProcessModel using RunConfig
# @tag('fixed_pt')
# class PySpkRecvModelFixed(PyLoihiProcessModel):
#     s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
#     spk_data: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

#     def run_spk(self):
#         """Receive spikes and store in an internal variable"""
#         spk_in = self.s_in.recv()
#         self.spk_data[self.time_step - 1, :] = spk_in

MAX_NUM =2000
ITERS =10


def preprocess_label(label,dim):
    labels = np.zeros((len(label),dim))
    for i in range(len(label)):
        labels[i] = label[i]


def generate_inputs(inputs,vth,T):
    res = np.zeros((T,len(inputs),inputs.shape[1]))
    for j in range(len(inputs)):
        input_ = inputs[j]
        intervals = (vth/input_).astype(int)+1
        for t in range(T):
            for i in range(len(input_)):
                if (t+1)%intervals[i] ==0:
                    res[t,j,i] = 1
    return res

def generate_spikes(num_samples,inputs,vth, T):
    res = generate_inputs(inputs,vth, T)
    tmp_res = res
    spikes = np.zeros([res.shape[2], res.shape[0]*res.shape[1]])
    print(spikes.shape)
    # for kk in range(len(spikes)):
    #     spikes[kk] = res [:,0,kk]
    for kk in range(len(spikes)):
        for i in range(num_samples):
                spikes[kk,i*T:(i+1)*T] = res[:,i,kk]
    
    return spikes


def init_weights( inputs, outputs, h, init=0):
    w_h = []
    w_o = []
    tmpp = np.random.normal(0, np.sqrt(3.0 / float(inputs)), [inputs, h[0]])
    if init == 1:
        cut = np.sqrt(3.0 / float(inputs)) * init
        tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
        tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(inputs))
        tmpp[tmpp > cut] = np.sqrt(3.0 / float(inputs))
    elif init == 2:
        if len(h) > 1:
            tmpp = np.random.normal(0, np.sqrt(6.0 / (float(inputs) + h[1])), [inputs, h[0]])
        else:
            tmpp = np.random.normal(0, np.sqrt(6.0 / (float(inputs) + outputs)), [inputs, h[0]])
    w_h.append(tmpp)
    for i in range(0, len(h) - 1):
        tmpp = np.random.normal(0, np.sqrt(3.0 / h[i]), [h[i], h[i + 1]])
        if init == 1:
            cut = np.sqrt(3.0 / float(h[i])) * init
            tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
            tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(h[i]))
            tmpp[tmpp > cut] = np.sqrt(3.0 / float(h[i]))
        elif init == 2:
            if (i + 2) < len(h):
                tmpp = np.random.normal(0, np.sqrt(6.0 / (h[i] + h[i + 2])), [h[i], h[i + 1]])
            else:
                tmpp = np.random.normal(0, np.sqrt(6.0 / (h[i] + outputs)), [h[i], h[i + 1]])
        w_h.append(tmpp)
    
    tmpp = np.random.normal(0, np.sqrt(3.0 / h[-1]), [h[-1], outputs])
    if init == 1:
        cut = np.sqrt(3.0 / float(h[-1])) * init
        tmpp[np.bitwise_and(tmpp > -cut, tmpp < cut)] = 0.0
        tmpp[tmpp < -cut] = -np.sqrt(3.0 / float(h[-1]))
        tmpp[tmpp > cut] = np.sqrt(3.0 / float(h[-1]))
    elif init == 2:
        tmpp = np.random.normal(0, np.sqrt(6.0 / h[-1]), [h[-1], outputs])
    w_o = tmpp

    return w_h, w_o


def Init_Threshold(inputs, outputs, h, threshold_h, threshold_o, init=0, dfa=0, norm=0.0):
    hiddenThr1 = threshold_h
    outputThr1 = threshold_o
    threshold_h = []
    # hThr1 = inputs*0.1
    hThr = inputs * np.sqrt(3.0 / float(inputs)) / (2.0)
    if init == 2:
        if len(h) > 1:
            hThr = inputs * np.sqrt(6.0 / (float(inputs) + h[1])) / 2.0
        else:
            hThr = inputs * np.sqrt(6.0 / (float(inputs) + outputs)) / 2.0

    threshold_h.append(hThr * hiddenThr1 / float(len(h) + 1))
    for i in range(len(h) - 1):
        if init != 2:
            threshold_h.append(h[i] * (np.sqrt(3.0 / h[i]) / 2.0) * hiddenThr1 / (len(h) - i - 0))
        elif init == 2:
            if (i + 2) < len(h):
                threshold_h.append(
                    h[i] * (np.sqrt(6.0 / (h[i] + h[i + 2])) / 2.0) * hiddenThr1 / (len(h) - i - 1))
            else:
                threshold_h.append(
                    h[i] * (np.sqrt(6.0 / (h[i] + outputs)) / 2.0) * hiddenThr1 / (len(h) - i - 1))

    if init != 2:
        threshold_o = h[-1] * (np.sqrt(3.0 / h[-1]) / 2.0) * outputThr1

    elif init == 2:
        threshold_o = h[-1] * (np.sqrt(6.0 / h[-1]) / 2.0) * outputThr1

    ethreshold_h = []
    ethreshold_o = []
    if init == 2:
        if len(h) > 1:
            hThr = h[1] * np.sqrt(6.0 / (float(inputs) + h[1])) / 2.0
        else:
            hThr = outputs * np.sqrt(6.0 / (float(inputs) + outputs)) / 2.0

        if norm == 0:
            ethreshold_h.append(hThr * hiddenThr1 / (len(h)))
        else:
            ethreshold_h.append(norm / (len(h)))

        for i in range(len(h) - 1):
            if (i + 2) < len(h):
                if norm == 0:
                    ethreshold_h.append(
                        h[i + 2] * (np.sqrt(6.0 / (h[i] + h[i + 2])) / 2.0) * hiddenThr1 / (i + 2))
                else:
                    ethreshold_h.append(norm / (i + 2))
            else:
                if norm == 0:
                    ethreshold_h.append(
                        outputs * (np.sqrt(6.0 / (h[i] + outputs)) / 2.0) * hiddenThr1 / (i + 2))
                else:
                    ethreshold_h.append(norm / (i + 2))

    if init != 2:
        ts = 1.0
        ehiddenThr1 = hiddenThr1
        eoutputThr1 = outputThr1
        for i in range(len(h)):
            if norm == 0:
                if i == len(h) - 1:

                    if dfa == 1:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))

                    elif dfa == 2:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                    else:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / h[-1]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                else:
                    if dfa == 1:
                        ethreshold_h.append(
                            h[i + 1] * (np.sqrt(3.0 / h[i + 1]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
                    elif dfa == 2:
                        ethreshold_h.append(
                            outputs * (np.sqrt(3.0 / outputs) / 2.0) * ehiddenThr1 / ((i + 2) * ts))

                    else:
                        ethreshold_h.append(
                            h[i + 1] * (np.sqrt(3.0 / h[i]) / 2.0) * ehiddenThr1 / ((i + 2) * ts))
            else:
                ethreshold_h.append(norm / (1))
    if init == 3:
        tss = 1.0
        ethreshold_h = np.divide(threshold_h, tss)
        ethreshold_o = np.divide(threshold_o, tss)

    return threshold_h, threshold_o, ethreshold_h, ethreshold_o
    
def to_integer(weights, bitwidth, normalize=True):
    """Convert weights and biases to integers.

    :param np.ndarray weights: 2D or 4D weight tensor.
    :param np.ndarray biases: 1D bias vector.
    :param int bitwidth: Number of bits for integer conversion.
    :param bool normalize: Whether to normalize weights and biases by the
        common maximum before quantizing.

    :return: The quantized weights and biases.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    max_val = np.max(np.abs(weights)) \
        if normalize else 1
    a_min = -2**bitwidth
    a_max = - a_min - 1
    weights = np.clip(weights / max_val * a_max, a_min, a_max).astype(int)
    return weights

class multipattern_learning:
    def __init__(self,dim,conv, time_steps ):
        self.w_h= []
        self.w_o =[]
        self.w_fixed =[]
        self.w_h_fixed = []
        self.dim = dim
        self.time_steps = time_steps
        self.conv = conv
        
 #online/offline accepting data
    def streaming(self, dataset,online=False):
        dim = self.dim
        self.threshold_h, self.threshold_o, self.ethreshold_h, self.ethreshold_o =Init_Threshold(inputs= self.dim[0],outputs=self.dim[-1],h =[self.dim[1]],threshold_h=0.5,threshold_o=0.1)
        data = dataset[0]
        labels = dataset[1]
        data = data[:MAX_NUM]
        labels = labels[:MAX_NUM]
        bs = int(len(data)/ITERS)
        if self.conv:
            model = ann_model()
            new_data = np.zeros((len(data),self.dim[0]))
            for kk in range(int(len(data/bs))):
                new_data[kk*bs:(kk+1)*bs] = model(data[bs*kk:(kk+1)*bs])
            data = new_data        
         
        data_index = (np.linspace(0, len(data) - 1, len(data))).astype(int)
        data = np.expand_dims(np.reshape(data, [len(data),self.dim[0]]), axis=0)
       
        if len(self.w_a) ==0:
            self.w_a, self.w_b = init_weights(inputs= self.dim[0], outputs=self.dim[-1],h=[self.dim[1]])
        else:
            self.w_a[0] = (np.transpose(self.w_h/np.max(self.w_h))).astype(float)
            self.w_b = (np.transpose(self.w_h/np.max(self.w_b))).astype(float)
        labels = np.argmax(labels,axis=-1)
        
        for i in range(10):   
            spikes = np.zeros([self.time_steps,bs, dim[0]]).astype(float)     
            tmp_rand = np.random.random([self.time_steps, 1, 1])
            randy = np.tile(tmp_rand, (1, bs, dim[0]))
            tmp_d = np.tile(data[:, data_index[i * bs:(i + 1) * bs], :], (self.time_steps, 1, 1))
            spikes = randy < (tmp_d)
            input_spikes = spikes.astype(float)
            label = labels[i*bs:(i+1)*bs]
            
            self.hiddens = [dim[1]]
            self.outputs = dim[-1]
            # for feed forward network
            n_hidden = len(self.hiddens)

    
            # hidden_spikes = np.zeros()
            hidden_spikes = [1] * n_hidden
            U_h = [1] * n_hidden
            
            for i in range(n_hidden):
                hidden_spikes[i] = np.zeros([self.time_steps, bs, self.hiddens[i]], dtype=bool)
            # hidden_spikes = np.zeros([hiddens,T])
            output_spikes = np.zeros([self.time_steps, bs, self.outputs], dtype=bool)
            for i in range(n_hidden):
                U_h[i] = np.zeros([self.time_steps, bs, self.hiddens[i]])
            U_o = np.zeros([self.time_steps, bs, self.outputs])
            


            # for feedback (error) network
            delta = np.zeros([self.time_steps, bs, self.outputs])
            delta_h = [1] * n_hidden
            for i in range(n_hidden):
                delta_h[i] = np.zeros([self.time_steps, bs, self.hiddens[i]])

            sdelta = np.zeros([self.time_steps, bs, self.outputs])
            sdelta_h = [1] * n_hidden
            for i in range(n_hidden):
                sdelta_h[i] = np.zeros([self.time_steps, bs, self.hiddens[i]])

            """
            
            """



            ## main training loop (main algorithm)
            for t in range(1, self.time_steps):
                # first phase
                U_h[0][t, :, :] = oe.contract("Bi,ij->Bj", input_spikes[t - 1, :, :], self.w_a[0])  + U_h[0][t - 1, :, :]
                hidden_spikes[0][t, :, :] = U_h[0][t, :, :] >= self.threshold_h[0]
                U_h[0][t, hidden_spikes[0][t, :, :]] = 0

                '''
                correct here
                '''
                # U_h[0] = U_h[0].astype(int)

            
                U_o[t, :, :] = oe.contract("Bi,ij->Bj", hidden_spikes[n_hidden - 1][t - 1, :, :], self.w_b) + U_o[t - 1, :, :] 

                output_spikes[t, :, :] = U_o[t, :, :] >= self.threshold_o
                U_o[t, output_spikes[t, :, :]] = 0

                for k in range(bs):
                    delta[t, k, :] = delta[t-1, k, :]
                    delta[t, k, label[k]] = delta[t - 1, k, label[k]] + (np.sum(output_spikes[t - 1, k, label[k]], axis=0) < 1).astype(float)
                    delta[t, k, np.concatenate((np.arange(0,label[k]), np.arange(label[k]+1,self.outputs)))] = delta[t-1, k, np.concatenate((np.arange(0,label[k]), np.arange(label[k]+1,self.outputs)))] -(np.sum(output_spikes[t - 1:t, k, np.concatenate((np.arange(0,label[k]), np.arange(label[k]+1,self.outputs)))], axis=0) >= 1).astype(float)


                    # generating spikes for the loss (error spikes at output layer)
            
                    self.ethreshold_o = 1
                    sdelta[t, delta[t, :, :] >= self.ethreshold_o] = 1
                    delta[t, delta[t, :, :] >= self.ethreshold_o] = 0
                    sdelta[t, delta[t, :, :] <= -self.ethreshold_o] = -1
                    delta[t, delta[t, :, :] <= -self.ethreshold_o] = 0
                    




                delta_h[n_hidden - 1][t, :, :] = oe.contract("Bj,ij->Bi", sdelta[t-1, :, :], self.w_b) + delta_h[n_hidden - 1][t - 1, :, :]


                sdelta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] >= self.ethreshold_h[n_hidden - 1]] = 1.0
                delta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] >= self.ethreshold_h[n_hidden - 1]] = 0.0
                sdelta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] <= -self.ethreshold_h[n_hidden - 1]] = -1.0
                delta_h[n_hidden - 1][t, delta_h[n_hidden - 1][t, :, :] <= -self.ethreshold_h[n_hidden - 1]] = 0.0




            tmp0 = (np.sum(hidden_spikes[n_hidden - 1][: self.time_steps, :, :],
                                    axis=0, keepdims=True))
            tmp1 = (1*np.sum((sdelta), axis=0, keepdims=True))
            tpp = np.mean(oe.contract("iBj,iBk->Bjk", tmp0, tmp1), axis=0) / float(self.time_steps)
            newlr = 0.003

            self.w_b +=  (np.multiply(tpp, newlr))

            tmp0 = (np.sum(input_spikes[ :self.time_steps, :, :], axis=0,
                                    keepdims=True))

            tmp1 = (np.sum((sdelta_h[0][:self.time_steps,:,:]), axis=0, keepdims=True) )



            tpp = np.mean(oe.contract("iBj,iBk->Bjk", tmp0, tmp1), axis=0) / float(self.time_steps)
 
            self.w_a[0] += np.multiply(tpp, newlr)
            
        self.w_h = np.transpose(to_integer(self.w_a[0],8))
        self.w_o = np.transpose(to_integer(self.w_b,8))
  



