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


input_layer = Input((32,32,1))

layer = Conv2D(filters=16, 
                kernel_size=(5, 5), 
                strides=(2, 2), 
                padding='valid',
                use_bias=False,
                input_shape=input_shape,
                activation='relu')(input_layer)

layer = Conv2D(filters=8, 
                kernel_size=(5, 5), 
                strides=(2, 2), 
                padding='valid',
                use_bias=False,
                input_shape=input_shape,
                activation='relu')(layer)

layer = Flatten()(layer)

layer = Dense(100,
            activation='relu',
            use_bias=False)(layer)
layer = Dense(num_classes,
            activation='softmax',
            use_bias=False)(layer)

ann_model = Model(input_layer, layer)
ann_model.load_weights("ann_wgt.h5")
parameters = ann_model.get_weights()
weights = parameters

conv_wgt_1 = to_integer(weights[0],8)#size 5,5,1,16 #out 14 ,14 ,16
conv_wgt_2 = to_integer(weights[1],8) #size 5,5,16,8 #out 5, 5, 8
conv_wgt_1 = np.reshape(conv_wgt_1,(conv_wgt_1.shape[-1],conv_wgt_1.shape[0],conv_wgt_1.shape[1],conv_wgt_1.shape[2]))
conv_wgt_2 = np.reshape(conv_wgt_2,(conv_wgt_2.shape[-1],conv_wgt_2.shape[0],conv_wgt_2.shape[1],conv_wgt_2.shape[2]))
# weight_dims = [
#         out_channels,
#         kernel_size[0], kernel_size[1],
#         in_channels // groups
# ]

input_shape_1 = (32,32,1)
input_shape_2 = (14,14,16)
conv1 = Conv(
        weight= conv_wgt_1,
        input_shape= input_shape_1,
        padding=(0,0),
        stride=(2,2)
      )
conv2 = Conv(
        weight= conv_wgt_2,
        input_shape= input_shape_2,
        padding=(0,0),
        stride=(2,2)
      )
lif_in = LIF(shape=input_shape_1, vth=50, du=0, dv=0,
          bias_mant=25)
lif_in.s_out.connect(conv1.in_ports.s_in)
conv1.out_ports.a_out.connect(conv2.in_ports.s_in)

lif_in.run(condition=RunSteps(num_steps=13000), run_cfg=Loihi2SimCfg(select_tag= "fixed_pt"))
lif_in.stop()

