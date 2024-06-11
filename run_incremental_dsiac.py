from incremental import train_incremental, memory_data, merge_set,prepare_new_tasks
from emstdp import loihi2_net
import numpy as np
DIM =[100,50,9]
T = 64


'''
Memory data
'''
#============================================================================
mem_data = np.load("./incremental_learning/mem_data.npy")
mem_label = np.load("./incremental_learning/mem_label.npy")
idx = np.arange(len(mem_data))
np.random.shuffle(idx)
mem_set = [mem_data[idx],mem_label[idx]]



#============================================================================


'''
Pretrained weights
'''
#============================================================================
w_h_pretrained = np.load("./incremental_learning/pre_trained_w_h.npy")[:,:,0]
w_o_pretrained = np.load("./incremental_learning/pre_trained_w_o.npy")
#============================================================================


'''
Prepare Task Data
'''
#============================================================================
task_1_label = [2,3]
task_2_label = [4,5]

#prepare data, use prepared data from yolo chips
data_path = "./incremental_learning/dsiac_chip.npy"
label_file = "./incremental_learning/dsiac_chip_labels.txt"

task_1_set= prepare_new_tasks(data_file = data_path, tasks_label = task_1_label,data_label= label_file,each_task = 100)
task_2_set = prepare_new_tasks(data_file = data_path, tasks_label = task_2_label,data_label= label_file,each_task = 100)


#============================================================================

''' ***********
Incremental  Training 

    ***********
'''
w_h, w_o = train_incremental(w_h = w_h_pretrained, w_o = w_o_pretrained, task_1 = task_1_set,\
                             task_2 = task_2_set, mem_set = mem_set ,epochs = 1 )


''' ***********
Incremental  Testing: on ALL dataset 

    ***********
'''
print("Start Testing!")
data = np.load("incre_data.npy")
labels = np.load("incre_label.npy")
dataset = [data[-100:],labels[-100:]]
net = loihi2_net(dim =DIM,time_steps = T,w_h = w_h,w_o=w_o)
net.test_loihi(dataset)








    