import numpy as np
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.proc.lif.process import LIF,LIFReset
from lava.proc.dense.process import Dense
from utils import init_weights
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

from lava.proc.monitor.process import Monitor
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
loihi2_is_available = Loihi2.is_loihi2_available

from lava.proc.dense.process import LearningDense, Dense, DelayDense
from lava.proc.lif.process import LIFReset
from lava.magma.core.run_configs import Loihi2HwCfg ,Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.proc.monitor.process import Monitor
import time
import typing
from lava.proc.conv.process import Conv
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


from utils import Init_Threshold
from utils import init_weights
from utils import generate_spikes
from utils import pre_process_test
# from lava.proc.ImgToSpk import InpImgToSpk,PyInpImgToSpkModel,InputAdapter,NxInputAdapter
from utils import transform


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

# class Edense(AbstractProcess):
#     """Dense connections between neurons.
#     Realizes the following abstract behavior:
#     a_out = W * s_in
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         shape = kwargs.get("shape", (1, 1))
#         self.s_in = InPort(shape=(shape[1],))
#         self.a_out = OutPort(shape=(shape[0],))
#         self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))
        
#     def set_weights(self,wgt):
#         self.weights = wgt


class loihi2_net(multipattern_learning):
    def __init__(self, dim,time_steps,w_h=[], w_o = [],threshold_h = 0.5,
                 conv_input_shape = (32,32,1), threshold_o = 0.1, 
                 conv_wgt= [], conv=False, fast_io = False,
                 loihi =True):
        super().__init__(dim = dim,conv = conv,w_h =w_h,w_o=w_o,time_steps = time_steps)
        
        #check initialization
        if not isinstance(dim , typing.List):
            print('the dimension should be a list with size 3.')
            
        
        
        self.dim = dim
        self.fast_io = fast_io
        self.fwd_nodes = dict()
        self.bwd_nodes = dict()
        self.fwd_con = dict()
        self.fwd_hidden_vth = []
        self.fwd_output_vth = []
        self.loihi_label = []
        self.loihi_neg_out = []
        self.loihi_pos_hid =[]
        self.loihi_neg_hid = []
        self.loihi_fwd_connections = dict()
        self.conv = conv
        self.conv_input_shape = conv_input_shape
        self.fast_io = fast_io

        self.pos_out = []
        self.pos_hidden = []
        self.pos_in = []

        self.hidden_weights = w_h
        self.out_weights= w_o

        temp_h, temp_o = init_weights(inputs= self.dim[0], outputs=self.dim[-1],h=[self.dim[1]])
        if len(w_h) ==0:
            self.w_h = np.transpose(temp_h[0])
            self.w_o = np.transpose(temp_o)
        else:
            self.w_h = w_h
            self.w_o = w_o
        self.time_steps = time_steps
        self.lr = 0.003
        self.loihi_fwd_nodes = dict()
        self.loihi_fwd_connections = dict()
        self.fac = 255
        '''
        dummy ones, used to check trace
        '''
        self.bw_w_h  = np.zeros((1,self.dim[1]))
        self.bw_w_o = np.transpose(self.w_o)
        a,b,self.eth_h,_ =Init_Threshold(inputs= self.dim[0],outputs=self.dim[-1],h =[self.dim[1]],threshold_h=threshold_h,threshold_o= threshold_o)
        self.fwd_hidden_vth = a[0]
        self.fwd_output_vth = b
        self.loihi = loihi
        self.bs = 5
        self.loihi_bwd_nodes = dict()
        

        
        if len(w_h)!=0:
            self.w_h = w_h
            self.w_o = w_o
            self.w_a[0] = np.transpose(w_h)
            self.w_b = np.transpose(w_o)
            
        if self.loihi:
            self.w_h = to_integer(self.w_h,8)
            self.w_o = to_integer(self.w_o,8)
            self.fwd_hidden_vth = (255*self.fwd_hidden_vth).astype(int)
            self.fwd_output_vth = (255*self.fwd_output_vth).astype(int)
            if self.conv:
                self.create_conv_loihi_network(conv_wgt[0],conv_wgt[1])
            else:
                self.create_loihi_dense_network()
                              
                self.create_loihi_error_path()
            
        else:  
            self.create_network()
            self.connect_fwd()
            self.connect_fwd_bwd()
            self.connect_bwd()
            


    def create_nodes(self, shape, du=1,dv=0,bias_mant =0, vth =1,loihi=False):
        if loihi:
            du = 4095
            dv = 0
            
        return LIF(
            shape = shape,
            du = du,
            dv = dv,
            vth = vth,
            bias_mant = bias_mant,
            # reset_interval= self.time_steps
        )
    '''
    create forward neural network
    '''
    def create_network(self):
        '''
        create nodes
        '''
        for i in range(len(self.dim)):
            if i == 0:
                self.fwd_nodes[i] = self.create_nodes(shape = (self.dim[i],), vth =1)
            elif i == 1:
                self.fwd_nodes[i] = self.create_nodes(shape = (self.dim[i],), vth =self.fwd_hidden_vth)
                self.bwd_nodes[i] = self.create_nodes(shape = (self.dim[i],), vth =self.eth_h)
            else:
                self.fwd_nodes[i] = self.create_nodes(shape = (self.dim[i],), vth =self.fwd_output_vth)
                # final output backward layer
                self.bwd_nodes[i] = self.create_nodes(shape = (self.dim[i],), vth = 1)
                # pos layer
        '''
        need a dummy connection here, from output layer to the label
        '''
        self.label = self.create_nodes(shape = (self.dim[2],), vth = 0.98)
        self.pos_hidden = self.create_nodes(shape =(self.dim[1],), vth = self.eth_h)

        
    def connect_fwd(self):
       
        lr =  Loihi2FLearningRule(
                dw = '0*x0-0*x0',
                x1_impulse = 1,
                x1_tau = 70000,
                y1_impulse = 1,
                y1_tau= 70000,
                y3_impulse = 1,
                y3_tau=4096,
                t_epoch =1 
            )


        for i in range(int(len(self.dim)-1)):
            if i == 0:
                self.fwd_con[i] = LearningDense(
                    weights = self.w_h,
                    learning_rule = lr
                )

            else:
                self.fwd_con[i] = LearningDense(
                    weights = self.w_o,
                    learning_rule = lr
                )
            
            self.fwd_nodes[i].s_out.connect(self.fwd_con[i].s_in)
            self.fwd_con[i].a_out.connect(self.fwd_nodes[i+1].a_in)
            # self.fwd_nodes[i+1].s_out.connect(self.fwd_con[i].s_in_bap)
    
    def create_loihi_error_path(self):
        self.label = LIF(shape =(self.dim[2],), vth=1, du=4095,dv=0)
        self.label_out_con = Dense(weights =np.eye(self.dim[2]))
        self.label.s_out.connect(self.label_out_con.s_in)
        self.label_out_con.a_out.connect(self.lif_dense_out.a_in)


        #neg out 
        self.loihi_neg_out = LIF(shape =(self.dim[2],), vth=1, du=4095,dv=0)
        self.loihi_neg_hid = LIF(shape =(self.dim[1],), vth=1, du=4095,dv=0)
        #connections
        self.neg_out_con = Dense(weights = np.eye(self.dim[2]))
        self.neg_hid_con = Dense(weights = np.transpose(self.w_o))
        self.neg_hid_hid = Dense(weights = np.eye(self.dim[1]))
        
        self.lif_dense_out.s_out.connect(self.neg_out_con.s_in)
        self.neg_out_con.a_out.connect(self.loihi_neg_out.a_in)
        self.loihi_neg_out.s_out.connect(self.neg_hid_con.s_in)
        self.neg_hid_con.a_out.connect(self.loihi_neg_hid.a_in)
        self.loihi_neg_hid.s_out.connect(self.neg_hid_hid.s_in)
        self.neg_hid_hid.a_out.connect(self.lif_dense_hid.a_in)
        
    def create_loihi_dense_network(self):
        lr = Loihi2FLearningRule(
        dw= '2^-7*u7*y1*x1-2^-7*u7*y0*x1',  
        x1_impulse=1,
        x1_tau= 240,
        y1_impulse=1,
        y1_tau= 240,
        t_epoch=1)
        if self.fast_io:
             #fast leakage since the inputs are always 0 or 1
            self.lif_input = LIF(shape =(self.dim[0],), vth=1, du=4095,dv=4095)
        else:
            print("Bias input")
            self.lif_input = LIF(shape =(self.dim[0],), vth=1, du=4095,dv= 0)
            
        self.lif_dense_hid = LIF(
            shape = (self.dim[1],), vth= self.fwd_hidden_vth, du=4095,dv=0)
        self.dense_hid = LearningDense(
            weights = self.w_h,
            learning_rule= lr
            
        )
        self.dense_out = LearningDense(
            weights= self.w_o,
            learning_rule= lr
        )
        self.lif_dense_out = LIF(
            shape =(self.dim[2],), vth= self.fwd_output_vth, du=4095,dv=0 
        )
        self.lif_input.s_out.connect(self.dense_hid.s_in)
        self.dense_hid.a_out.connect(self.lif_dense_hid.a_in)
        self.lif_dense_hid.s_out.connect(self.dense_out.s_in)
        self.dense_out.a_out.connect(self.lif_dense_out.a_in)
    '''
    conv_wgt: 2 layer convolutional weights
    '''
    def create_conv_loihi_network(self, conv_wgt_1, conv_wgt_2,input_shape = (32,32,1)):
        shape = input_shape
        convwgt1 = np.reshape(conv_wgt_1,(conv_wgt_1.shape[-1],conv_wgt_1.shape[0],conv_wgt_1.shape[1],conv_wgt_1.shape[2]))           
        convwgt2 = np.reshape(conv_wgt_2,(conv_wgt_2.shape[-1],conv_wgt_2.shape[0],conv_wgt_2.shape[1],conv_wgt_2.shape[2]))
        lr = Loihi2FLearningRule(
        dw= '2^-6*u7*y1*x1-2^-7*u7*y0*x1',  
        x1_impulse=1,
        x1_tau= 240,
        y1_impulse=1,
        y1_tau= 240,
        t_epoch=1)
        self.lif_input = LIF(shape =(32,32,1), vth=1, du=4095,dv=0)
        self.conv_1 = Conv(
                weight = convwgt1,
                input_shape= self.conv_input_shape,
                padding=(0,0),
                stride=(2,2),
                )
        self.lif_inter = LIF(shape =self.conv_1.output_shape, vth=1, du=4095,dv=0)
        self.conv_2 = Conv(
                weight = convwgt2,
                input_shape= self.conv_1.output_shape,
                padding=(0,0),
                stride=(2,2),
                )
        # print("conv 1 out",self.conv_1.output_shape )
        # print("conv2 out", self.conv_1.output_shape)
        #dense in shape 200       
        self.lif_dense_in = LIF(shape =self.conv_2.output_shape, vth=1, du=4095,dv=0)
        # hidden size is 100
        self.dense_hid = LearningDense(
            weights = self.w_h,
            learning_rule= lr
            
        )
        self.lif_dense_hid = LIF(
            shape = (self.dim[1],), vth= self.fwd_hidden_vth, du=4095,dv=0
        )
        self.dense_out = LearningDense(
            weights= self.w_o,
            learning_rule= lr
        )
        self.lif_dense_out = LIF(
            shape =(self.dim[2],), vth= self.fwd_output_vth, du=4095,dv=0
        )
        print("out shape", self.conv_2.output_shape)
        self.lif_input.s_out.connect(self.conv_1.s_in)
        self.conv_1.a_out.connect(self.lif_inter.a_in)
        self.lif_inter.s_out.connect(self.conv_2.s_in)
        self.conv_2.a_out.connect(self.lif_dense_in.a_in)
        # # fully connected layers
        self.lif_dense_in.s_out.flatten().connect(self.dense_hid.s_in)
        self.dense_hid.a_out.connect(self.lif_dense_hid.a_in)
        self.lif_dense_hid.s_out.connect(self.dense_out.s_in)
        self.dense_out.a_out.connect(self.lif_dense_out.a_in)

     

    def test_conv_loihi_network(self, dataset, conv_wgt_1, conv_wgt_2, w_h,w_o, input_shape = (28,28,1)):
        convwgt1 = np.reshape(conv_wgt_1,(conv_wgt_1.shape[-1],conv_wgt_1.shape[0],conv_wgt_1.shape[1],conv_wgt_1.shape[2]))           
        convwgt2 = np.reshape(conv_wgt_2,(conv_wgt_2.shape[-1],conv_wgt_2.shape[0],conv_wgt_2.shape[1],conv_wgt_2.shape[2]))
    
        lif_input = LIF(shape =input_shape, vth=1000, du=4095,dv=0, name = "lif_input")
        conv_1 = Conv(
                weight = convwgt1,
                input_shape= input_shape,
                padding=(0,0),
                stride=(2,2),
                weight_exp =2,
                name = "conv1"
                )
        lif_inter = LIF(shape =conv_1.output_shape, vth=1000, du=4095,dv=0, name = "lif_inter",reset_interval = self.time_steps)
        conv_2 = Conv(
                weight = convwgt2,
                input_shape= conv_1.output_shape,
                padding=(0,0),
                weight_exp =2,
                stride=(2,2),
                name = "conv2"
                )

        lif_dense_in = LIFReset(shape =conv_2.output_shape, vth=1, du=4095,dv=0, name = "lif_dense_in",reset_interval = self.time_steps)
        # hidden size is 100
        dense_hid = Dense(
            weights = w_h,
            name = "dense_hid"
            
        )
        lif_dense_hid = LIFReset(
            shape = (self.dim[1],), vth= self.fwd_hidden_vth, du=4095,dv=0, name ="lif_dense_hid",reset_interval = self.time_steps
        )
        dense_out = Dense(
            weights= w_o,
            name = "dense_out"
        )
        dummy_conn = Dense(
            weights = np.eye(self.dim[2]),
            name = "dummy_conn"
        )
        out_conn = Dense(
            weights = np.eye(self.dim[2]),
            name = "out_conn"
        )
      
        lif_dense_out = LIFReset(
            shape =(self.dim[2],), vth= self.fwd_output_vth, du=4095,dv=0,
            name = "lif_dense_out",reset_interval = self.time_steps
        )
        
        c = LIF(shape = (self.dim[2],), vth =7000, du =4095, dv = 0, name = "c")
        assist = LIF(shape = (self.dim[2],), vth =2, du =4095, dv = 0, name = "assist")
        
        imgs = dataset[0]
        labels = dataset[1]
        demo_assist = self.create_bias(labels)


                
      

        lif_input.s_out.connect(conv_1.s_in)
        conv_1.a_out.connect(lif_inter.a_in)
        lif_inter.s_out.connect(conv_2.s_in)
        conv_2.a_out.connect(lif_dense_in.a_in)
        # fully connected layers
        lif_dense_in.s_out.flatten().connect(dense_hid.s_in)
        dense_hid.a_out.connect(lif_dense_hid.a_in)
        lif_dense_hid.s_out.connect(dense_out.s_in)
        dense_out.a_out.connect(lif_dense_out.a_in)
        

        
        lif_dense_out.s_out.connect(out_conn.s_in)
        out_conn.a_out.connect(c.a_in)
        assist.s_out.connect(dummy_conn.s_in)
        dummy_conn.a_out.connect(c.a_in)
        final_res = np.zeros((len(imgs), self.dim[2]))
        lif_input.run(condition=RunSteps(num_steps=1), run_cfg= Loihi2HwCfg(select_tag = "fixed_pt"))
        
        for i in range(10):
            lif_input.bias_mant.set((65*imgs[i]).astype(int))
            # pattern_pre.bias_mant.set(inputs[i])
            self.assist_test(assist,demo_assist[i])  
            lif_input.run(condition=RunSteps(num_steps= 32), run_cfg= Loihi2HwCfg(select_tag = "fixed_pt"))
            # lif_input.run(condition=RunSteps(num_steps= 32), run_cfg= Loihi2HwCfg())
            print(np.count_nonzero(lif_inter.v.get()))
            result = c.v.get()
            final_res[i] = result
            c.v.set(np.zeros([self.dim[2]]))
            lif_input.v.set(np.zeros(input_shape))
            assist.v.set(np.zeros([self.dim[2]]))
    
    
        lif_input.stop()
        print(np.argmax(final_res,axis =-1)[:20])
        # print('Testing Result:',np.sum(np.argmax(final_res,axis =-1)==np.argmax(labels,axis =-1))/num_samples)
        count = 0
        for i in range(len(final_res)):
            if len(np.array(labels).shape)>1:
                if np.argmax(final_res[i],axis =-1) == np.argmax(labels[i],axis =-1):
                    count+=1
            else:
                if np.argmax(final_res[i],axis =-1) ==  labels[i]:
                    count+=1
            
        print("acc is ", count/len(final_res))     
        '''
        reset convolutional layers and fully connected layers internal states
        If there's LIFReset available, use LIFReset instead
        '''
    def reset_conv_states(self):
        self.lif_input.vars.v.set(np.zeros(self.conv_input_shape))
        self.lif_inter.vars.v.set(np.zeros([self.conv_1.output_shape]))
        self.lif_dense_in.vars.v.set(np.zeros([self.conv_2.output_shape]))
        self.lif_dense_hid.vars.v.set(np.zeros([self.dim[1]]))
        self.lif_dense_out.vars.v.set(np.zeros([self.dim[2]]))
        
        
        
    '''
    Train loihi network for 1 epoch
    '''
    def train_loihi_network(self, dataset, w_h=[], w_o=[], fast_io = False):
        data = dataset[0]
        labels = dataset[1]
        if len(data[0].shape)>2:
            data = transform(data)
        #start = time.time()
        aux = dataset
        if not self.loihi:
            run_cfg = Loihi2SimCfg(select_tag= "fixed_pt")
        else:
            run_cfg = Loihi2HwCfg()

        if not self.fast_io:
            if not self.conv:
                self.lif_input.run(condition=RunSteps(num_steps=1), run_cfg=run_cfg)
                for i in range(len(data)):
                    inputs = data[i]
                    label = labels[i]
                    self.lif_input.bias_mant.set((65*inputs).astype(int))
                    #run intervals
                    self.lif_input.run(condition=RunSteps(num_steps=self.time_steps), run_cfg=  Loihi2HwCfg())
                    self.lif_input.v.set(np.zeros([self.dim[0]]))
            else:
                print("train conv network")
         
         
        else:
            generator_dense = Dense(
                weights = 2*np.eye(self.dim[0])
            )
            num_steps = self.time_steps
            num_samples = len(data)
            inp_data = self.generate_spikes(num_samples= num_samples,inputs= data,vth=1,
                                T= self.time_steps)
            inp_shape = generator_dense.s_in.shape
            generator = io.source.RingBuffer(data=inp_data)
            inp_adapter = eio.spike.PyToNxAdapter(shape=inp_shape)
            generator.s_out.connect(inp_adapter.inp)
            inp_adapter.out.connect(generator_dense.s_in)
            generator_dense.a_out.connect(self.lif_input.a_in)
            
            self.lif_input.run(condition=RunSteps(num_steps=num_steps*num_samples),
                  run_cfg=Loihi2HwCfg())
            
        self.lif_input.stop()
        self.streaming(aux)  
        end = time.time()
        #print("elapsed time: ", end - start)
        
        return self.w_h, self.w_o
        
    
    def reset_training_states(self):
        if not self.conv:
            self.lif_input.v.set(np.zeros([self.dim[0]]))
        else:
            self.lif_input.v.set(np.zeros(self.conv_input_shape))
        

    def connect_bwd(self):
        lr =  Loihi2FLearningRule(
                dw = '0*x0 -0*x0',
                x1_impulse = 1,
                x1_tau = 70000,
                y1_impulse = 1,
                y1_tau=70000,
                y3_impulse = 1,
                y3_tau=4096,
                t_epoch =1 
            )


        self.bwd_con = LearningDense(weights = self.bw_w_o, learning_rule = lr, name = "bwd_conn")
        self.bwd_nodes[2].s_out.connect(self.bwd_con.s_in)
        self.bwd_con.a_out.connect(self.bwd_nodes[1].a_in)
        # self.bwd_nodes[1].s_out.connect(self.bwd_con.s_in_bap)
        
                
        self.bwd_dummy_node = LIF(shape=(1,),du=1,dv=0)
        self.bwd_dummy_conn = LearningDense(weights=np.ones((1,self.dim[1])), learning_rule=lr)
        self.bwd_nodes[1].s_out.connect(self.bwd_dummy_conn.s_in)
        self.bwd_dummy_conn.a_out.connect(self.bwd_dummy_node.a_in)


        self.label_hid_con = LearningDense(weights = self.bw_w_o, learning_rule = lr, name = "label_hid_con")
        self.label.s_out.connect(self.label_hid_con.s_in)
        self.label_hid_con.a_out.connect(self.pos_hidden.a_in)
        
        self.label_dummy_node = LIF(shape=(1,),du=1,dv=0)
        self.label_dummy_conn = LearningDense(weights=np.ones((1,self.dim[1])), learning_rule=lr)
        self.pos_hidden.s_out.connect(self.label_dummy_conn.s_in)
        self.label_dummy_conn.a_out.connect(self.label_dummy_node.a_in)
        #self.pos_hidden.s_out.connect(self.label_hid_con.s_in_bap)
        
        dummy_con= Dense(weights = 0*np.eye(self.dim[2]))
        self.fwd_nodes[2].s_out.connect(dummy_con.s_in)
        dummy_con.a_out.connect(self.label.a_in)


    def assist_test(self,nodes,inputs):
        nodes.bias_mant.set(inputs.astype(int))
        
    def reset_traces(self):
        self.bwd_con.vars.x1.set(np.zeros((self.dim[2],)))
        self.bwd_dummy_conn.x1.set(np.zeros((self.dim[1],)))
        
        self.label_hid_con.vars.x1.set(np.zeros((self.dim[2],)))
        self.label_dummy_conn.vars.x1.set(np.zeros((self.dim[1],)))


        self.fwd_con[0].vars.x1.set(np.zeros((self.dim[0],)))
        self.fwd_con[1].vars.x1.set(np.zeros((self.dim[1],)))
        
    def create_bias(self,labels):
        demo_assist = np.zeros((len(labels),self.dim[2]))
        for i in range(len(labels)):
            if len(np.array(labels).shape)>1:
                demo_assist[i,int(np.argmax(labels[i]))] = 65
            else:
                demo_assist[i,int(labels[i])] = 65
        return demo_assist
        
    def test_loihi(self,dataset, w_h=[],w_o=[]):
        data = dataset[0]
        label = dataset[1]
        vth = 1
        vth_hid = 550
        vth_out = 210
        data = pre_process_test(data)
        T = self.time_steps
        num_samples = len(data)
        features = self.dim[0]
        b_features = self.dim[1]
        c_features = self.dim[2]
        self.conv= False
        inputs = data
        labels = label
        demo_assist = np.zeros((len(labels),c_features))
        
        for i in range(len(labels)):
            if len(np.array(labels).shape)>1:
                demo_assist[i,int(np.argmax(labels[i]))] = 65
            else:
                demo_assist[i,int(labels[i])] = 65
                
        # pattern_pre = RingBuffer(data=spikes.astype(int))
        # pattern_pre = LIFReset(shape=(features,),vth = 1,bias_mant= 0,du=4095,dv=0,reset_interval=64)
        pattern_pre = LIF(shape=(features,),vth = 1,bias_mant= 0,du=4095,dv=0)
        #w_h = w_h[0]
        if len(w_h) ==0:
            w_h = np.transpose(self.w_h)
            w_o = np.transpose(self.w_o)
        else:
            w_h = np.transpose(w_h)
            w_o = np.transpose(w_o)


        # a = LIFReset(shape=(b_features,),vth = vth_hid,bias_mant= 0,du=4095,dv=0,reset_interval=64)
        # b = LIFReset(shape=(c_features,),vth = vth_out,bias_mant= 0,du=4095,dv=0,reset_interval=64)
        if not self.loihi:
            a = LIF(shape=(b_features,),vth = vth_hid,bias_mant= 0,du=4095,dv=0)
            b = LIF(shape=(c_features,),vth = vth_out,bias_mant= 0,du=4095,dv=0)
            c = LIF(shape=(c_features,),vth = 1000,bias_mant= 0,du=4095,dv=0)

            con1 = Dense(weights= np.transpose(w_h))
            con = Dense(weights= np.transpose(w_o) )
            con2 = Dense(weights = np.eye(c_features))
            
            pattern_pre.s_out.connect(con1.s_in)
            con1.a_out.connect(a.a_in) 
            a.s_out.connect(con.s_in)
            con.a_out.connect(b.a_in)
            b.s_out.connect(con2.s_in)
            con2.a_out.connect(c.a_in)
            spk_prob = Monitor()
            spk_prob.probe(b.s_out,1+num_samples*T)
            pattern_pre.run(condition=RunSteps(num_steps= 1), run_cfg= Loihi2SimCfg(select_tag = "fixed_pt"))
            for i in range(num_samples):
                pattern_pre.bias_mant.set((65*inputs[i]).astype(int))
                # pattern_pre.bias_mant.set(inputs[i])
                pattern_pre.run(condition=RunSteps(num_steps= T), run_cfg= Loihi2SimCfg(select_tag = "fixed_pt"))
                pattern_pre.v.set(np.zeros([features]))
                b.vars.v.set(np.zeros([c_features]))
                a.v.set(np.zeros([b_features]))
                result = c.v.get()
                print(np.argmax(result))
                c.vars.v.set(np.zeros([c_features]))
                pattern_pre.pause()
            spks = spk_prob.get_data()[b.name]['s_out'][1:]
            pattern_pre.stop()
            res_spk = np.zeros([num_samples,c_features])
            for i in range(num_samples):
                res_spk[i] = np.sum(spks[i*T:(i+1)*T],axis =0)
            print("=========")    
            # print(out_spk)
            # print(np.argmax(res_spk)[:10])
            print(np.argmax(res_spk,axis= -1)[:10])
            # print(np.argmax(labels[:10],axis =-1))
            print(np.sum(np.argmax(res_spk,axis =-1)==np.argmax(labels,axis =-1))/num_samples)
            print("=========")
        else:
            start = time.time()
            a = LIF(shape=(b_features,),vth = vth_hid,bias_mant= 0,du=4095,dv=0)
            b = LIF(shape=(c_features,),vth = vth_out,bias_mant= 0,du=4095,dv=0)
            c = LIF(shape=(c_features,),vth = 7000,bias_mant= 0,du=4095,dv=0)
            assist = LIF(shape=(c_features,),vth = 1,bias_mant= 0,du=4095,dv=0)

            con1 = Dense(weights= np.transpose(w_h))
            con = Dense(weights= np.transpose(w_o) )
            con2 = Dense(weights = np.eye(c_features))
            con3 = Dense(weights = np.eye(c_features))
            dummy_conn = Dense(weights = 0*np.eye(c_features))
            
            pattern_pre.s_out.connect(con1.s_in)
            con1.a_out.connect(a.a_in) 
            a.s_out.connect(con.s_in)
            con.a_out.connect(b.a_in)
            b.s_out.connect(con2.s_in)
            con2.a_out.connect(c.a_in)
            assist.s_out.connect(con3.s_in)
            con3.a_out.connect(c.a_in)
            # c.s_out.connect(dummy_conn.s_in)
            # dummy_conn.a_out.connect(assist.a_in)


            final_res = np.zeros((num_samples, c_features))
            pattern_pre.run(condition=RunSteps(num_steps=1), run_cfg= Loihi2HwCfg())
            for i in range(num_samples):
                pattern_pre.bias_mant.set((65*inputs[i]).astype(int))
                # pattern_pre.bias_mant.set(inputs[i])
                self.assist_test(assist,demo_assist[i])
                pattern_pre.run(condition=RunSteps(num_steps= T), run_cfg= Loihi2HwCfg())
                pattern_pre.v.set(np.zeros([features]))
                b.vars.v.set(np.zeros([c_features]))
                a.v.set(np.zeros([b_features]))
                result = c.v.get()
                # print(assist.v.get())
                final_res[i] = result
                c.vars.v.set(np.zeros([c_features]))
                assist.v.set(np.zeros([c_features]))
                # pattern_pre.pause()

            pattern_pre.stop()
            # print(np.argmax(final_res,axis =-1)[:20])
            # print('Testing Result:',np.sum(np.argmax(final_res,axis =-1)==np.argmax(labels,axis =-1))/num_samples)
            count = 0
            for i in range(len(final_res)):
                if len(np.array(labels).shape)>1:
                    if np.argmax(final_res[i],axis =-1) == np.argmax(labels[i],axis =-1):
                        count+=1
                else:
                    if np.argmax(final_res[i],axis =-1) ==  labels[i]:
                        count+=1
                        
            end = time.time()
            print("acc is ", count/len(final_res))
            # print("elapsed time", end -start)
            return count/len(final_res)
        
    def test_non_conv_loihi_fast_io(self,dataset,w_h=[],w_o=[]):
         
        start = time.time()
        data = dataset[0]
        label = dataset[1]
        if len(w_h) ==0:
            weight = self.w_h
            weight1 = self.w_o
        else:
            weight = w_h
            weight1 = w_o
            # weight = 255*np.ones((100,200))
            # weight1 = 255*np.ones((10,100))
        # weight = 1*np.ones((100,200))
        # weight1 = 1*np.ones((10,100))
        num_steps = self.time_steps
        num_samples = len(data)

        
        dense = Dense(weights=weight)
        dense2 = Dense(weights=weight1)
        sigma = LIFReset(shape=dense.a_out.shape,dv=0, du= 4095, bias_mant=0, vth = 755,reset_interval =self.time_steps)
        lif = LIFReset(shape = (weight1.shape[0],), dv =0, du = 4095, bias_mant =0,vth =220,reset_interval = self.time_steps)
        inp_shape = dense.s_in.shape
        out_shape = lif.s_out.shape
        inp_data = self.generate_spikes(num_samples= num_samples,inputs= data,vth=1,
                                      T= self.time_steps)
        print("non zero data",np.count_nonzero(inp_data))

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
            final_res[i,:] = np.sum(out_data[:,(i*self.time_steps):(i+1)*self.time_steps],axis = -1)
        final_res = np.argmax(final_res, axis= -1)
        print(final_res)
        acc = (np.sum(final_res == np.argmax(label,axis=-1))/num_samples).astype(float) 
        print("Testing result is: ", acc)
        end = time.time()
        print("elapsed time", end -start)

       
    def connect_fwd_bwd(self):
        fwd_bwd_conn = Dense(
            weights = np.eye(self.dim[2]),
            name = "fwd_bwd_conn"
        )
        self.fwd_nodes[2].s_out.connect(fwd_bwd_conn.s_in)
        fwd_bwd_conn.a_out.connect(self.bwd_nodes[2].a_in)

        lr =  Loihi2FLearningRule(
        dw = '0*x0 -0*x0',
        x1_impulse = 1,
        x1_tau = 70000,
        y1_impulse = 1,
        y1_tau=70000,
        y3_impulse = 1,
        y3_tau=4096,
        t_epoch =1 
            )

    def generate_spikes(self,num_samples,inputs,vth, T):
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


    '''
    (1,bs.dim)
    '''
    def update(self,tmp_delta_o,tmp_delta_h,tmp_h_trace,tmp_input_trace):
        w_h = self.fwd_con[0].vars.weights.get()
        w_o = self.fwd_con[1].vars.weights.get()
        tpp_h =  np.mean(oe.contract("iBj,iBk->Bjk",tmp_input_trace , tmp_delta_h), axis=0) / float(self.time_steps)
        tpp_o =  np.mean(oe.contract("iBj,iBk->Bjk",tmp_h_trace , tmp_delta_o), axis=0) / float(self.time_steps)
        delta_h = np.multiply(tpp_h, self.lr)
        delta_h = np.clip(delta_h,-0.4,0.4)
        w_h+=  np.transpose(delta_h)
        delta_o = np.clip(np.multiply(tpp_o, self.lr),-0.5,0.5)
        w_o+= np.transpose(delta_o)
        self.fwd_con[0].vars.weights.set(w_h)
        self.fwd_con[1].vars.weights.set(w_o)
            
    def generate_inputs(self, inputs,vth):
        T = self.time_steps
        res = np.zeros((T,len(inputs),inputs.shape[1]))
        for j in range(len(inputs)):
            input_ = inputs[j]
            intervals = (vth/input_).astype(int)+1
            for t in range(T):
                for i in range(len(input_)):
                    if (t+1)%intervals[i] ==0:
                        res[t,j,i] = 1
        return res

    def fit(self,dataset):
        data = dataset[0]
        label = dataset[1]
        j = 0 
        tmp_delta_o = np.zeros((1,self.bs,self.dim[2]))
        tmp_delta_h = np.zeros((1,self.bs,self.dim[1]))
        tmp_h_trace = np.zeros((1,self.bs,self.dim[1]))
        tmp_input_trace = np.zeros((1,self.bs,self.dim[0]))
        #start run time
        self.fwd_nodes[0].run(condition=RunSteps(num_steps=1), run_cfg= Loihi2SimCfg(select_tag = "floating_pt"))
        # self.fwd_nodes[0].pause()
        for i in range(int(len(data))):
            # set label as bias
            bias =np.array(label[i])
            self.fwd_nodes[0].vars.bias_mant.set(data[i])
            self.label.bias_mant.set(bias)
            self.fwd_nodes[0].run(condition=RunSteps(num_steps=self.time_steps), run_cfg= Loihi2SimCfg(select_tag = "floating_pt"))
            

            if (i!=0) and (i%self.bs)==0:
                self.update(tmp_delta_o,tmp_delta_h,tmp_h_trace,tmp_input_trace)
                tmp_delta_o = np.zeros((1,self.bs,self.dim[2]))
                tmp_delta_h = np.zeros((1,self.bs,self.dim[1]))
                tmp_h_trace = np.zeros((1,self.bs,self.dim[1]))
                tmp_input_trace = np.zeros((1,self.bs,self.dim[0]))
                j = 0
            
            # size output
            delta_out_target = self.label_hid_con.vars.x1.get() - self.bwd_con.vars.x1.get()

            #size hidden layer
            delta_hid_target = self.label_dummy_conn.vars.x1.get()- self.bwd_dummy_conn.vars.x1.get()

            # size hidden layer
            hid_traces = self.fwd_con[1].vars.x1.get()
            
            input_traces = self.fwd_con[0].vars.x1.get()
            
            tmp_delta_o[0,j,:] = delta_out_target
            tmp_delta_h[0,j,:] = delta_hid_target

            tmp_h_trace[0,j,:] = hid_traces
            tmp_input_trace[0,j,:] = input_traces

            j = j+1

            #update
            self.reset_traces()
            self.fwd_nodes[0].pause()
        self.hidden_weights = self.fwd_con[0].vars.weights.get()
        self.out_weights  = self.fwd_con[1].vars.weights.get()
        self.fwd_nodes[0].stop()
        
        return self.hidden_weights, self.out_weights



def test_non_conv_loihi_fast_io(dataset,w_h=[],w_o=[],time_steps =32):
    data = dataset[0]
    label = dataset[1]

    weight = w_h
    weight1 = w_o
    num_steps = time_steps
    num_samples = len(data)

    
    dense = Dense(weights=weight)
    dense2 = Dense(weights=weight1)
    sigma = LIFReset(shape=dense.a_out.shape,dv=0, du= 4095, bias_mant=0, vth = 755,reset_interval =time_steps)
    lif = LIFReset(shape = (weight1.shape[0],), dv =0, du = 4095, bias_mant =0,vth =220,reset_interval = time_steps)
    inp_shape = dense.s_in.shape
    out_shape = lif.s_out.shape
    inp_data = generate_spikes(num_samples= num_samples,inputs= data,vth=1,
                                  T= time_steps)
    print("non zero data",np.count_nonzero(inp_data))

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
    acc = (np.sum(final_res == np.argmax(label,axis=-1))/num_samples).astype(float) 
    print("Testing result is: ", ACC)    

