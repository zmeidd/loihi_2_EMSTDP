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
dim =[200,100,9]
w_h = np.load("./weights/w_h.npy")
w_o = np.load("./weights/w_o.npy")
net = loihi2_net(dim = dim,w_h = w_h, w_o = w_o, time_steps = 32)

print(labels)
data = preprocess_raw_imgs(files, file_path)
print(data.shape)
net.test_loihi([data[:10],labels[:10]],w_h = w_h, w_o = w_o)
