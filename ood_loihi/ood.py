# import logging
# import numpy as np
# import os
# from typing import Tuple
# from lava.lib.dl import netx
# from lava.magma.core.decorator import implements, requires
# from lava.magma.core.model.py.model import PyLoihiProcessModel
# from lava.magma.core.model.py.ports import PyInPort, PyOutPort
# from lava.magma.core.model.py.type import LavaPyType
# from lava.magma.core.process.ports.ports import InPort, OutPort
# from lava.magma.core.process.process import AbstractProcess
# from lava.magma.core.process.variable import Var
# from lava.magma.core.resources import CPU
# from lava.magma.core.run_conditions import RunSteps
# from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
# from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
# from lava.proc.dense.process import Dense
# from lava.proc.lif.process import LIF, LIFReset
# from scipy.special import softmax
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from skimage.transform import resize
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# logging.disable(logging.WARNING)
# import numpy as np
# from model import model

# def to_integer(weights, bitwidth, normalize=True):
#     """Convert weights and biases to integers.

#     :param np.ndarray weights: 2D or 4D weight tensor.
#     :param np.ndarray biases: 1D bias vector.
#     :param int bitwidth: Number of bits for integer conversion.
#     :param bool normalize: Whether to normalize weights and biases by the
#         common maximum before quantizing.

#     :return: The quantized weights and biases.
#     :rtype: tuple[np.ndarray, np.ndarray]
#     """

#     max_val = np.max(np.abs(weights)) \
#         if normalize else 1
#     a_min = -2**bitwidth
#     a_max = - a_min - 1
#     weights = np.clip(weights / max_val * a_max, a_min, a_max).astype(int)
#     return weights



# class InputAdapter(AbstractProcess):
#   """
#   Input Adapter Process.
#   """
#   def __init__(self, shape: Tuple[int, ...]):
#     super().__init__(shape=shape)
#     self.inp = InPort(shape=shape)
#     self.out = OutPort(shape=shape)

# @implements(proc=InputAdapter, protocol=LoihiProtocol)
# @requires(CPU)
# class PyInputAdapter(PyLoihiProcessModel):
#   """
#   Input adapter model for CPU, i.e., when your spike input process is on CPU and
#   you plan to send the input spikes to a Loihi2 Simulation running on CPU.
#   """
#   inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
#   out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

#   def run_spk(self):
#     self.out.send(self.inp.recv())


# class ood_spk(AbstractProcess):
#   """
#   Input process to convert flattened images to binary spikes.
#   """
#   def __init__(self, out_shape, dataset, curr_img_id, v_thr=1, n_steps = 100):
#     super().__init__()
#     self.spk_out = OutPort(shape=out_shape, )
#     self.dataset = Var(shape= dataset.shape, init= dataset)
#     self.n_ts = Var(shape=(1, ), init=n_steps)
    
#     self.inp_img = Var(shape=out_shape, )
#     self.v = Var(shape=out_shape, init=0)
#     self.vth = Var(shape=(1, ), init=v_thr)
#     self.curr_img_id = Var(shape=(1, ), init=curr_img_id)
    
# @implements(proc=ood_spk, protocol=LoihiProtocol)
# @requires(CPU)
# class PyInpImgToSpkModel(PyLoihiProcessModel):
#   """
#   Python implementation for the above `InpImgToSpk` process.
#   """
#   spk_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
#   curr_img_id: int = LavaPyType(int, int, precision=32)
#   n_ts: int = LavaPyType(int, int, precision=32)
#   '''
#   executing arrays
#   '''
#   v: np.ndarray = LavaPyType(np.ndarray, float)
#   vth: float = LavaPyType(float, float)
#   dataset: np.ndarray = LavaPyType(np.ndarray, float)
#   inp_img: np.ndarray = LavaPyType(np.ndarray, float)


#   def __init__(self, proc_params):
#     super().__init__(proc_params=proc_params)
#     self.net = model()

#   def post_guard(self):
#     """
#     Guard function for post-management phase, necessary to update the next image
#     index after the current image is processed.

#     Note: The execution control calls `post_guard()` after `run_spk()` every
#     time-step, before updating the `self.time_step` variable to next time-step.
#     """
#     if self.time_step % self.n_ts == 1: # n_ts steps passed, one image processed.
#       return True

#     return False

#   def run_post_mgmt(self):
#     """
#     Post-management phase executed only when the above `post_guard()` returns
#     True -> then, move to the next image, reset the neuron states, etc.
#     """
#     x = self.net(np.expand_dims(self.dataset[self.curr_img_id],axis=0))
#     self.inp_img = x
#     self.v = np.zeros(self.v.shape, dtype=float)
#     self.curr_img_id += 1

#   def run_spk(self):
#     """
#     Spiking phase, this is executed every simulation time-step unconditionally,
#     and first in order of all the phases.
#     """
#     if self.time_step % self.n_ts == 1:
#      self.inp_img = np.zeros(self.inp_img.shape, dtype=float)
#      self.v = np.zeros(self.v.shape, dtype=float)

#     J = self.inp_img 
#     self.v[:] = self.v[:] + J[:]
#     mask = self.v >=self.vth
#     self.v[mask] = 0
#     self.spk_out.send(mask)    



# w_h = to_integer(np.load("w_h_oe.npy"),8)  
# w_o = to_integer(np.load("w_o_oe.npy"),8)
# print(w_h.shape)
# print(w_o.shape)

# T = 32
# #input adapter
# # inp_adp = InputAdapter(shape=(200,))
# # con_test = Dense(weights= 64*np.eye(200))

# lif_test = LIF(shape=(100,), vth = 100, du =4095, dv=0)
# lif_1 = LIF(shape=(50,), vth= 552, du =4095,dv =0)
# lif_2 = LIF(shape=(9,), vth= 156, du =4095, dv=0)
# lif = LIF(shape=(9,), vth= 10000, du =4095, dv =0)

# con_1 = Dense(weights= np.transpose(w_h))
# con_2 = Dense(weights= np.transpose(w_o))
# con_3 = Dense(weights= np.eye(9))





# # con_1 = Dense(weights= np.transpose(np.ones((200,100))))
# # con_2 = Dense(weights= np.transpose(np.ones((100,9))))
# # con_3 = Dense(weights= np.eye(9))

# net = model()
# dataset = net(np.load("out_data.npy")[:20]).numpy()
# # # prc = ood_spk(out_shape=(200,),dataset= dataset,curr_img_id=0, n_steps=T)
# # # prc.spk_out.connect(inp_adp.inp)
# # # inp_adp.out.connect(con_test.s_in)
# # # con_test.a_out.connect(lif_test.a_in)
# lif_test.s_out.connect(con_1.s_in)
# con_1.a_out.connect(lif_1.a_in)
# lif_1.s_out.connect(con_2.s_in)
# con_2.a_out.connect(lif_2.a_in)
# lif_2.s_out.connect(con_3.s_in)
# con_3.a_out.connect(lif.a_in)
# #inp_adp.out.connect(con_1.s_in)
# # con_1.a_out.connect(lif_1.a_in)

# # # lif_1.s_out.connect(con_2.s_in)
# # # con_2.a_out.connect(lif_2.a_in)

# # # lif_2.s_out.connect(con_3.s_in)
# # # con_3.a_out.connect(lif.a_in)


# print(dataset.shape)


# outs = []

# lif_test.run(condition=RunSteps(num_steps= 1), run_cfg= Loihi2SimCfg(select_tag= "fixed_pt"))
# for i in range(20):
#     lif_test.bias_mant.set((65*100*dataset[i]).astype(int))
#     lif_test.run(condition=RunSteps(num_steps= T), run_cfg= Loihi2SimCfg(select_tag= "fixed_pt"))

#     if len(outs) == 0:
#         outs = lif.v.get()//64
#     else:
#         outs = np.vstack((outs,lif.v.get()//64))
        
#     lif_test.v.set(np.zeros((100,)))
#     lif_1.v.set(np.zeros((50,)))
#     lif_2.v.set(np.zeros((9,)))
#     lif.v.set(np.zeros((9,)))
# lif_test.stop()

# in_score = softmax(outs/T, axis =-1)
# in_score = -np.max(in_score, axis=1)
# np.save("in_score.npy", in_score)


import logging
import numpy as np
import os
from typing import Tuple
from lava.lib.dl import netx
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF, LIFReset
from scipy.special import softmax
from utils import get_measures
from utils import print_measures
from skimage.transform import resize

import numpy as np
from model import model
from utils import get_measures

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



class InputAdapter(AbstractProcess):
  """
  Input Adapter Process.
  """
  def __init__(self, shape: Tuple[int, ...]):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapter(PyLoihiProcessModel):
  """
  Input adapter model for CPU, i.e., when your spike input process is on CPU and
  you plan to send the input spikes to a Loihi2 Simulation running on CPU.
  """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())


class ood_spk(AbstractProcess):
  """
  Input process to convert flattened images to binary spikes.
  """
  def __init__(self, out_shape, dataset, curr_img_id, v_thr=1, n_steps = 100):
    super().__init__()
    self.spk_out = OutPort(shape=out_shape, )
    self.dataset = Var(shape= dataset.shape, init= dataset)
    self.n_ts = Var(shape=(1, ), init=n_steps)
    
    self.inp_img = Var(shape=out_shape, )
    self.v = Var(shape=out_shape, init=0)
    self.vth = Var(shape=(1, ), init=v_thr)
    self.curr_img_id = Var(shape=(1, ), init=curr_img_id)
    
@implements(proc=ood_spk, protocol=LoihiProtocol)
@requires(CPU)
class PyInpImgToSpkModel(PyLoihiProcessModel):
  """
  Python implementation for the above `InpImgToSpk` process.
  """
  spk_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
  curr_img_id: int = LavaPyType(int, int, precision=32)
  n_ts: int = LavaPyType(int, int, precision=32)
  '''
  executing arrays
  '''
  v: np.ndarray = LavaPyType(np.ndarray, float)
  vth: float = LavaPyType(float, float)
  dataset: np.ndarray = LavaPyType(np.ndarray, float)
  inp_img: np.ndarray = LavaPyType(np.ndarray, float)


  def __init__(self, proc_params):
    super().__init__(proc_params=proc_params)
    self.net = model()

  def post_guard(self):
    """
    Guard function for post-management phase, necessary to update the next image
    index after the current image is processed.

    Note: The execution control calls `post_guard()` after `run_spk()` every
    time-step, before updating the `self.time_step` variable to next time-step.
    """
    if self.time_step % self.n_ts == 1: # n_ts steps passed, one image processed.
      return True

    return False

  def run_post_mgmt(self):
    """
    Post-management phase executed only when the above `post_guard()` returns
    True -> then, move to the next image, reset the neuron states, etc.
    """
    x = self.net(np.expand_dims(self.dataset[self.curr_img_id],axis=0))
    self.inp_img = x
    self.v = np.zeros(self.v.shape, dtype=float)
    self.curr_img_id += 1

  def run_spk(self):
    """
    Spiking phase, this is executed every simulation time-step unconditionally,
    and first in order of all the phases.
    """
    if self.time_step % self.n_ts == 1:
     self.inp_img = np.zeros(self.inp_img.shape, dtype=float)
     self.v = np.zeros(self.v.shape, dtype=float)

    J = self.inp_img 
    self.v[:] = self.v[:] + J[:]
    mask = self.v >=self.vth
    self.v[mask] = 0
    self.spk_out.send(mask)    



w_h = to_integer(np.load("./files/w_h_oe.npy"),8)  
w_o = to_integer(np.load("./files/w_o_oe.npy"),8)




'''
Outlier detectector,
can be implemented on both software implementation and hardware implementation
'''
def detect_ood(data_set,  w_h =w_h, w_o = w_o, shape = 100):

    T = 32
    net = model()
    data_set = net(data_set).numpy()
    # lif_test = LIF(shape=(100,), vth = 10, du =4095, dv=0)
    lif_test = LIF(shape=(100,), vth = 10, du =4095, dv=0)
    lif_1 = LIF(shape=(50,), vth= 552, du =4095,dv =0)
    lif_2 = LIF(shape=(9,), vth= 156, du =4095, dv=0)
    lif = LIF(shape=(9,), vth= 10000, du =4095, dv =0)
    
    # inp_adp = InputAdapter(shape=(shape,))
    con_1 = Dense(weights= np.transpose(w_h))
    con_2 = Dense(weights= np.transpose(w_o))
    con_3 = Dense(weights= np.eye(9))

    # prc = ood_spk(out_shape=(shape,),dataset= data_set,curr_img_id=0, n_steps=T)
    # prc.spk_out.connect(inp_adp.inp)
    # inp_adp.out.connect(lif_test.a_in)
    
    lif_test.s_out.connect(con_1.s_in)
    con_1.a_out.connect(lif_1.a_in)
    lif_1.s_out.connect(con_2.s_in)
    con_2.a_out.connect(lif_2.a_in)
    lif_2.s_out.connect(con_3.s_in)
    con_3.a_out.connect(lif.a_in)
    
    
    out = []
    lif_test.run(condition=RunSteps(num_steps= 1), run_cfg= Loihi2SimCfg(select_tag= "fixed_pt"))
    #lif_test.run(condition=RunSteps(num_steps= 1), run_cfg= Loihi2HwCfg())
    print("Start OOD Testing")
    #prc.run(condition=RunSteps(num_steps= 1), run_cfg= Loihi2SimCfg(select_tag= "fixed_pt"))
    for i in range(len(data_set)):
        lif_test.bias_mant.set((65*10*data_set[i]).astype(int))
        #lif_test.run(condition=RunSteps(num_steps= T), run_cfg= Loihi2HwCfg())
        lif_test.run(condition=RunSteps(num_steps= T), run_cfg= Loihi2SimCfg(select_tag= "fixed_pt"))
        #prc.run(condition=RunSteps(num_steps= T), run_cfg= Loihi2SimCfg(select_tag= "fixed_pt"))
        if len(out) == 0:
            out = (lif.v.get())//64
        else:
            out = np.vstack((out, (lif.v.get())//64))
        lif_test.v.set(np.zeros((100,)))
        lif_1.v.set(np.zeros((50,)))
        lif_2.v.set(np.zeros((9,)))
        lif.v.set(np.zeros((9,)))
        lif_test.pause()
    lif_test.stop()
    
    outs = 2*out/T
    in_score = softmax(outs, axis= -1)
    in_score = -np.max(in_score, axis=1)
    return in_score
    

def detector(dataset):
    scores = detect_ood(dataset)
    th = np.load("./files/ood_treshold.npy")

    return scores <= th
    
# np.save("ood_treshold.py", threshold)
# auroc_list, aupr_list, fpr_list = [], [], []

# def get_and_print_results(in_score, out_score, num_to_avg= 1):

#     aurocs, auprs, fprs = [], [], []
#     for _ in range(num_to_avg):
#         measures = get_measures(out_score, in_score)
#         aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

#     auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
#     auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
#     print_measures(auroc, aupr, fpr, "DSIAC")


# get_and_print_results(in_score, out_score)