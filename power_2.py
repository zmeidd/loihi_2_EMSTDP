# import os
# import sys
# import logging
# import unittest

# import numpy as np
# from tests.lava.test_utils.utils import Utils

# # process definition
# from lava.magma.core.process.process import AbstractProcess
# from lava.magma.core.process.ports.ports import InPort, OutPort

# # models definition
# from lava.magma.core.decorator import implements, requires
# from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
# from lava.magma.core.model.sub.model import AbstractSubProcessModel
# from lava.magma.core.model.py.ports import PyInPort, PyOutPort
# from lava.magma.core.model.py.type import LavaPyType

# # in built processes
# from lava.proc.lif.process import LIF,LIFReset
# from lava.proc.sdn.process import Sigma
# from lava.proc.dense.process import Dense
# from lava.proc.conv.process import Conv
# from lava.proc import io
# from lava.proc import embedded_io as eio

# # execution
# from lava.magma.core.run_configs import Loihi2HwCfg
# from lava.magma.core.run_conditions import RunSteps
# import time

# num_steps = 32
# samples = 100
# weight0 = np.ones((100,200))
# weight1 = np.ones((10,100))
# dense = Dense(weights=weight0)
# dense2 = Dense(weights=weight1)
# sigma = LIFReset(shape=dense.a_out.shape, dv=0,du = 4095, vth =1, reset_interval = 32)
# lif = LIFReset(shape = dense2.a_out.shape,dv =0 , du =4095, vth =1 ,reset_interval = 32)
# inp_shape = dense.s_in.shape
# out_shape = lif.s_out.shape
# print("inP_shape", inp_shape)
# print("out shape", out_shape)

# inp_data = np.ones((inp_shape[0], num_steps*100))
# generator = io.source.RingBuffer(data=inp_data)
# inp_adapter = eio.spike.PyToNxAdapter(shape=inp_shape)
# out_adapter = eio.spike.NxToPyAdapter(shape=out_shape)
# logger = io.sink.RingBuffer(shape=out_shape, buffer=num_steps*samples)

# generator.s_out.connect(inp_adapter.inp)
# inp_adapter.out.connect(dense.s_in)
# dense.a_out.connect(sigma.a_in)
# sigma.s_out.connect(dense2.s_in)
# dense2.a_out.connect(lif.a_in)
# lif.s_out.connect(out_adapter.inp)
# out_adapter.out.connect(logger.a_in)

# start = time.time()

# sigma.run(condition=RunSteps(num_steps=num_steps*samples),
#           run_cfg=Loihi2HwCfg())
# out_data = logger.data.get().astype(np.int16)
# sigma.stop()

# end = time.time()
# print(out_data.shape)
# print("elapsed time:", end -start)




import os
import sys
import logging
import unittest

import numpy as np
from tests.lava.test_utils.utils import Utils

# process definition
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

# models definition
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# in built processes
from lava.proc.lif.process import LIF,LIFReset
from lava.proc.dense.process import Dense
from lava.proc.conv.process import Conv
from lava.proc import io
from lava.proc import embedded_io as eio

# execution
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
import time


from tests.lava.test_utils.utils import Utils

# process definition
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

# models definition
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# in built processes
from lava.proc.lif.process import LIF,LIFReset
from lava.proc.dense.process import Dense
from lava.proc.conv.process import Conv
from lava.proc import io
from lava.proc import embedded_io as eio

import time
from utils import generate_spikes

num_steps = 32
##
200*200
200*200
##
# weight = np.eye(200) * 10
# weight1 = np.ones((10,200))
# dense = Dense(weights=weight)
# dense2 = Dense(weights=weight1)
# sigma = LIFReset(shape=dense.a_out.shape)
# lif = LIFReset(shape = (10,))
# inp_shape = dense.s_in.shape
# out_shape = lif.s_out.shape
# inp_data = np.ones((inp_shape[0], num_steps*100))
# print("input shape", inp_shape)
# print("out shape", out_shape)


# generator = io.source.RingBuffer(data=inp_data)
# inp_adapter = eio.spike.PyToNxAdapter(shape=inp_shape)
# out_adapter = eio.spike.NxToPyAdapter(shape=out_shape)
# logger = io.sink.RingBuffer(shape=out_shape, buffer=num_steps*100)

# generator.s_out.connect(inp_adapter.inp)
# inp_adapter.out.connect(dense.s_in)

# dense.a_out.connect(sigma.a_in)
# sigma.s_out.connect(dense2.s_in)
# dense2.a_out.connect(lif.a_in)
# lif.s_out.connect(out_adapter.inp)
# out_adapter.out.connect(logger.a_in)

# start = time.time()

# sigma.run(condition=RunSteps(num_steps=num_steps),
#           run_cfg=Loihi2HwCfg())
# out_data = logger.data.get().astype(np.int16)
# sigma.stop()

# print("out_data shape", out_data.shape)
# end = time.time()
# print("elapsed time:", end -start)




num_steps = 32
##
200*200
200*200
##
num_samples = 1000


data_train = np.load("x_train.npy")
data_label = np.load("y_train.npy")

data = data_train[:1000]
label =data_label[:1000]


time_steps = 32
weight = np.load("w_h.npy")
weight1 = np.load("w_o.npy")
dense = Dense(weights=weight)
dense2 = Dense(weights=weight1)
sigma = LIFReset(shape=dense.a_out.shape,dv=0, du= 4095, vth =3*255, reset_interval = num_steps)
lif = LIFReset(shape = (weight1.shape[0],), dv =0, du = 4095, vth =int(255*0.86) ,reset_interval = num_steps)
inp_shape = dense.s_in.shape
out_shape = lif.s_out.shape
# inp_data = np.ones((inp_shape[0], num_steps*num_samples))
inp_data = generate_spikes(num_samples= num_samples,inputs= data,vth=1,
                                  T= 32)


generator = io.source.RingBuffer(data=inp_data)
inp_adapter = eio.spike.PyToNxAdapter(shape=inp_shape)
out_adapter = eio.spike.NxToPyAdapter(shape=out_shape)
logger = io.sink.RingBuffer(shape=out_shape, buffer=num_steps*num_samples)

generator.s_out.connect(inp_adapter.inp)
inp_adapter.out.connect(dense.s_in)

dense.a_out.connect(sigma.a_in)
sigma.s_out.connect(dense2.s_in)
dense2.a_out.connect(lif.a_in)
lif.s_out.connect(out_adapter.inp)
out_adapter.out.connect(logger.a_in)

start = time.time()

sigma.run(condition=RunSteps(num_steps=num_steps*num_samples),
          run_cfg=Loihi2HwCfg())
out_data = logger.data.get().astype(np.int16)
sigma.stop()
final_res = np.zeros((num_samples, 10))
out_data_1 = np.zeros_like(out_data)
out_data_1[:,0:-1] = out_data[:,1:] 
for i in range(num_samples):
    final_res[i,:] = np.sum(out_data[:,(i*time_steps):(i+1)*time_steps],axis = -1)
final_res = np.argmax(final_res, axis= -1)
print(final_res)
acc = (np.sum(final_res == np.argmax(label,axis=-1))/num_samples).astype(float) 
print("Testing result is: ", acc)   
print("out_data shape", out_data.shape)
end = time.time()
print("elapsed time:", end -start)

















