import numpy as np

import logging
import numpy as np
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
# import pyximport; pyximport.install()
import logging, os
from lava.proc.lif.process import LIF,LIFReset
from lava.proc.dense.process import Dense
logging.disable(logging.WARNING)
from lava.magma.core.model.py.model import PyLoihiProcessModel
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pickle
import time
import random
import numpy as np
from lava.magma.core.process.variable import Var
import numpy as np
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from typing import Tuple
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.proc import embedded_io as eio
from lava.proc.ImgToSpk import InpImgToSpk
from utils import loihi_model
from utils import loihi_model
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import sys
import inspect
def transfer_dataset(d_set,time_steps = 32):
    num_samples = int(len(d_set)/time_steps)
    res = np.zeros([num_samples,d_set.shape[1]])
    for i in range(num_samples):
        res[i] = np.sum(d_set[i*time_steps:(i+1)*time_steps,:],axis = 0)

    return res
        



x = np.load("imgs.npy")[:2]
# net = loihi_model()
# res = net(x)

data = x
start = time.time()
inp = InpImgToSpk(curr_img_id =0 , data = x)
for i in range(2):
    inp.run(condition=RunSteps(num_steps=32), run_cfg = Loihi2SimCfg(select_tag = "fixed_pt"))
    
# train_set = inp.img_set()
# res = transfer_dataset(train_set)
inp.stop()
print("elapsed time", time.time() -start)








# def test_conv_loihi(w_h,w_o,T=32,num_samples = 100):
#     b_features = 100
#     c_features = 10
#     vth_hid = 300
#     vth_out = 100

#     labels =np.load("loihi_label.npy")[:num_samples]
#     a = LIFReset(shape=(b_features,),vth = vth_hid,bias_mant= 0,du=4095,dv=0,reset_interval= T)
#     b = LIFReset(shape=(c_features,),vth = vth_out,bias_mant= 0,du=4095,dv=0,reset_interval= T)
#     c = LIFReset(shape=(c_features,),vth = 1000,bias_mant= 0,du=4095,dv=0,reset_interval = T)
#     con1 = Dense(weights= w_h)
#     con = Dense(weights= w_o)
#     con2 = Dense(weights = np.eye(c_features))

#     img_to_spk = InpImgToSpk(img_shape=(1,32,32,1), n_tsteps= T, curr_img_id=0)
#     # inp_adp = InputAdapter(shape= (200,))
#     img_to_spk.spk_out.connect(con1.s_in)
#     con1.a_out.connect(a.a_in) 
#     a.s_out.connect(con.s_in)
#     con.a_out.connect(b.a_in)
#     b.s_out.connect(con2.s_in)
#     con2.a_out.connect(c.a_in)

#     start = time.time()
#     final_res = np.zeros((num_samples, c_features))
#     for i in range(num_samples):
#         img_to_spk.run(
#               condition=RunSteps(num_steps=32), run_cfg = Loihi2SimCfg(
#                   select_tag = "fixed_pt",# To select fixed point implementation.
#                 exception_proc_model_map={
#                     #InpImgToSpk: PyInpImgToSpkModel,
#                     # OutSpkToCls: PyOutSpkToClsModel,
#                     # InputAdapter: NxInputAdapter,
#                     # OutputAdapter: NxOutputAdapter
#                     })
#         )
#         result = c.v.get()
#         print(result)
#         final_res[i] = result
#     print("final_res", np.argmax(final_res,axis = -1)[:10])
#     print(labels[:10])
#     print(labels.shape)
#     print("elapsed time",time.time()-start)  

# # label = np.load("loihi_label.npy")[:3000]
# # labels = np.zeros((3000,10))
# # for i in range(3000):
# #     labels[i,np.argmax(label[i])] =1
# # dataset = [np.load("loihi_data.npy")[:3000],labels]
# # net = loihi2_net([200,100,10],time_steps = 32, conv =True, fast_io= False)
# # w_h,w_o = net.streaming(dataset)
# # print("done")
# # np.save("w_h.npy",w_h)
# # np.save("w_o.npy",w_o)


# w_h = np.load("w_h.npy")
# w_o = np.load("w_o.npy")
# test_conv_loihi(w_h,w_o)
# # print("elapsed time:", time.time()-start)