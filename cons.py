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
import os
import numpy as np
from lava.proc import io
from lava.utils.system import Loihi2
from lava.proc.lif.process import LIF
from lava.proc.conv.process import Conv
from lava.proc.dense.process import Dense   
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from keras.models import Model
from keras.layers import Dropout, Flatten, Conv2D, Input, MaxPooling2D, Dense, AveragePooling2D
import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.proc.lif.process import LIF,LIFReset
from lava.proc.dense.process import Dense
from utils import init_weights
import numpy as np
import opt_einsum as oe
from lava.utils.system import Loihi2
from lava.proc import io
from lava.proc import embedded_io as eio
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore')
from lava.magma.core.run_conditions import RunSteps
from lava.proc.dense.process import Dense
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule

from utils import multipattern_learning
from utils import generate_inputs
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.process import LIF, LIFReset
from utils import Init_Threshold
from utils import init_weights
os.environ["SLURM"] = '1'
# os.environ['PYTHONBUFFERED'] = '1'
# Loihi2.preferred_partition = 'oheogulch'
# loihi2_is_available = Loihi2.is_loihi2_available



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

num_classes = 10
input_shape = (32,32,1)
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
    
# def train(e):
#     data_label = np.load("loihi_label.npy")[:2000]
#     label = data_label[:2000]
#     x =np.load("loihi_data.npy")[:2000]
#     data_set = [x, label]
#     net = multipattern_learning(dim = [200,100,10],conv=True, time_steps = 32)
#     for j in range(e):
#         w_h, w_o = net.streaming(data_set)
#         print("done")
        
#     np.save("w_h.npy",w_h)
#     np.save("w_o.npy",w_o)


from utils import simple_softmax_conv_model
from utils import loihi_model
#get weights
ann_model = simple_softmax_conv_model(10)
ann_model.load_weights("cnn.h5")
wgt = ann_model.get_weights()


conv_wgt_1 = wgt[0]
conv_wgt_2 = wgt[1]

conv_wgt_1 = np.reshape(conv_wgt_1,(conv_wgt_1.shape[-1],conv_wgt_1.shape[0],conv_wgt_1.shape[1],conv_wgt_1.shape[2]))
conv_wgt_2 = np.reshape(conv_wgt_2,(conv_wgt_2.shape[-1],conv_wgt_2.shape[0],conv_wgt_2.shape[1],conv_wgt_2.shape[2]))


conv_wgt_1= to_integer(conv_wgt_1,8)
conv_wgt_2= to_integer(conv_wgt_2,8)

input_shape_1 = (32,32,1)
input_shape_2 = (14,14,16)


data = np.ones((32,32,1,32))
import time
conv_1 = Conv(
    weight = conv_wgt_1,
    input_shape= input_shape_1,
    padding=(0,0),
    stride=(2,2),
    weight_exp = 1
    )

print("conv_wgt max ", np.max(conv_wgt_1))
conv_2 = Conv(
        weight = conv_wgt_2,
        input_shape= conv_1.output_shape,
        padding=(0,0),
        stride=(2,2),
        weight_exp =1,
        )
w_h = np.load("w_h.npy")
w_o = np.load("w_o.npy")
w_h = to_integer(w_h,8)
w_o = to_integer(w_o,8)
data_train = np.load("imgs.npy")[:2000]
data_label = np.load("label.npy")[:2000]
print("data_train shape", data_train.shape)

data = data_train[:2000]
label = data_label[:2000]
data = np.ones((1,32,32,1))

#connections layers for convolutional layers
lif_input = LIF(shape =(32,32,1), vth=1, du=4095,dv=0)
lif_inter = LIF(shape = conv_1.output_shape, vth=1, du=4095,dv=0)


        #dense in shape 200       
lif_dense_in = LIF(shape =conv_2.output_shape, vth=1, du=4095)
# hidden size is 100
dense_hid = Dense(
    weights = w_h,
)
lif_dense_hid = LIF(
    shape = (100,), vth= 765, du=4095,dv=0
)
dense_out = Dense(
    weights= w_o,
)
lif_dense_out = LIF(
    shape =(10,), vth=  200, du=4095,dv=0
)

c = LIF(shape=(10,),vth = 1000,bias_mant= 0,du=4095,dv=0, reset_interval = 32)
con2 = Dense(weights = np.eye(10))

d = LIF(shape=(200,),vth = 10000,bias_mant= 0,du=4095,dv=0)
con3 = Dense(weights = np.eye(200))


lif_input.s_out.connect(conv_1.s_in)
conv_1.a_out.connect(lif_inter.a_in)
lif_inter.s_out.connect(conv_2.s_in)
conv_2.a_out.connect(lif_dense_in.a_in)
lif_dense_in.s_out.flatten().connect(con3.s_in)
con3.a_out.connect(d.a_in)


res = np.zeros((2000,200))
lif_input.run(condition=RunSteps(num_steps=1), run_cfg= Loihi2SimCfg(select_tag = "fixed_pt"))

for i in range(len(data)):
    inputs = data[i]
    lif_input.bias_mant.set((65*inputs).astype(int))
    # pattern_pre.bias_mant.set(inputs[i])
    lif_input.run(condition=RunSteps(num_steps= 32), run_cfg= Loihi2SimCfg(select_tag = "fixed_pt"))
    lif_input.v.set(np.zeros((32,32,1)))
    lif_dense_in.v.set(np.zeros((200,)))
    lif_inter.v.set(np.zeros(conv_1.output_shape))
    res[i] = d.v.get()/(32*64)
    d.v.set(np.zeros((200,)))
lif_input.stop()

np.save("conv_data.npy", res)

# # # fully connected layers
# lif_dense_in.s_out.flatten().connect(dense_hid.s_in)
# # lif_dense_in.s_out.flatten().connect(con3.s_in)
# dense_hid.a_out.connect(lif_dense_hid.a_in)
# # con3.a_out.connect(d.a_in)
# lif_dense_hid.s_out.connect(dense_out.s_in)
# dense_out.a_out.connect(lif_dense_out.a_in)
# lif_dense_out.s_out.connect(con2.s_in)
# con2.a_out.connect(c.a_in)
# start = time.time()



# '''
# Train one epoch
# '''
# num_samples = len(data)
# tmp_out = np.zeros((num_samples, 200))
# final_res = np.zeros((num_samples, 10))
# lif_input.run(condition=RunSteps(num_steps=1), run_cfg= Loihi2SimCfg(select_tag = "fixed_pt"))
# for i in range(len(data)):
#     inputs = data[i]
#     lif_input.bias_mant.set((65*inputs).astype(int))
#     # pattern_pre.bias_mant.set(inputs[i])
#     lif_input.run(condition=RunSteps(num_steps= 32), run_cfg= Loihi2SimCfg(select_tag = "fixed_pt"))
    
#     lif_input.v.set(np.zeros((32,32,1)))
#     lif_dense_out.vars.v.set(np.zeros([10]))
#     lif_dense_hid.v.set(np.zeros([100]))
#     # tmp_out[i] = d.v.get().flatten()/2000
#     lif_dense_in.v.set(np.zeros(conv_2.output_shape))
#     lif_inter.v.set(np.zeros(conv_1.output_shape))
#     # d.v.set(np.zeros([200]))
#     result = c.v.get()
#     c.v.set(np.zeros([10]))
  
#     # print(result)
#     final_res[i] = result
#     #lif_input.pause()
# lif_input.stop()
# print('Testing Result:',np.sum(np.argmax(final_res,axis =-1)==label)/100)

# np.save("conv_train.npy",tmp_out)
# end = time.time()
# print("elapsed time", end -start)