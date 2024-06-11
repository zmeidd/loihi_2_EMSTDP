import os 
file_path = os.getcwd()+ "/test_imgs"
files = os.listdir(file_path)
class_name = {'2S3':0,  'MP2':1, 'DM2':2, 'R70':3, 'TLB':4, 'kup':5, 'ort':6,  'T72':7, '3-4':8}
labels = []
for i in range(len(files)):
    labels.append(int(class_name[files[i][-7:-4]]))

from emstdp import loihi2_net
import numpy as np
from utils import preprocess_raw_imgs
dim =[100,50,9]
w_h = np.load("./weights/w_h.npy")[:,:,0]
w_o = np.load("./weights/w_o.npy")
print(w_o)
net = loihi2_net(dim = dim,w_h = [], w_o = [], time_steps = 64)

data = np.load("demo_data.npy")
labels = np.load("demo_label.npy")
print(labels)
#net.test_loihi([data[:1200],labels[:1200]],w_h = w_h, w_o = w_o)
for i in range(3):
    w_h, w_o = net.train_loihi_network([data[:300],labels[:300]],w_h = w_h, w_o = w_o)
    del net
    net = loihi2_net(dim = dim,w_h = w_h, w_o = w_o, time_steps = 64)
    net.test_loihi([data[:100],labels[:100]],w_h = w_h, w_o = w_o)

