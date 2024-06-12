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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from emstdp import loihi2_net

NUM_TASKS = 2
DIM = [100,50,9]
TIME_STEPS= 64
TASK_2 = [2,3]
TASKS_3 = [4,5]
'''
200/tasks for old classes
'''
class_name = {'2S3':0,  'BMP2':1, 'BRDM2':2, 'BTR70':3, 'MTLB':4, 'PICKUP':5, 'SPORT':6,  'T72':7, 'ZSU23-4':8}
pre_trained = [0,1,6,7,8]
task_2 = [2,3]
task_3 = [4,5]

'''
merge memory set and current task's dataset,
shuffle the index of the merged dataset
'''
def merge_set(mem_set, cur_set):
    merged = [np.vstack((cur_set[0], mem_set[0])), np.append(np.array(cur_set[1]),np.array(mem_set[1]))]
    arr = np.arange(len(merged[0]))
    np.random.shuffle(arr)
    merged = [merged[0][arr], merged[1][arr]]
    
    return merged



class emstdp_results:
    def __init__(self, testing_task, testing_sample, training_sample, accuracy):
        self.training_task = testing_task
        self.testing_sample = testing_sample
        self.accuracy = accuracy
        self.training_sample = training_sample
    def __str__(self):
        return f"Current training task:{self.training_task}\nNumber of Training Samples:{self.training_sample:.1f}\nNumber of Testing Samples:{self.testing_sample}\nAccuracy: {self.accuracy:.2f}"

def print__results(results):
    print("\nIncremental learning Results:")
    for result in results:
        print("-" * 30)
        print(result)
    print("-" * 30)


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # 

def get_tank_name(task):
    name_list = []
    class_name = {'2S3':0,  'BMP2':1, 'BRDM2':2, 'BTR70':3, 'MTLB':4, 'Pickup':5, 'Sport':6,  'T72':7, 'ZSU23-4':8}
    for value in task:
        name_list.append(get_key_from_value(dictionary= class_name, value=value))
    return name_list



def memory_data(data,labels,sub_label, k =50):
    new_data = np.zeros((int(len(sub_label)*k),32,32,1))
    new_label = []
    count = 0
    idx = 0
    for i in range(len(sub_label)):
        key = sub_label[i]
        for m in range(len(data)):
            if (labels[m] == key) and (idx - (i+1)*k)!=0 :
                new_data[idx] = data[m]
                new_label.append(labels[m])
                idx+=1
        
    


        
    if len(new_label)!= len(new_data):
        print("wrong dimension: dim data! =dim label")

        
    return new_data, new_label




def prepare_new_tasks(data_file, data_label, tasks_label, each_task = 150, train_size = 2000):

   
    class_name = {'2S3':0,  'BMP2':1, 'BRDM2':2, 'BTR70':3, 'MTLB':4, 'Pickup':5, 'Sport':6,  'T72':7, 'ZSU23-4':8}
    training_set = []
    label_name = []
    with open(data_label) as file:
        for line in file:
            label_name.append(line.rstrip())
    
    raw_data =np.load(data_file)
    train_size = 10000
    
    class_name = {'2S3':0,  'BMP2':1, 'BRDM2':2, 'BTR70':3, 'MTLB':4, 'PICKUP':5, 'SPORT':6,  'T72':7, 'ZSU23-4':8}
    labels = []
    count = 0
    data = np.zeros((train_size,32,32,1))
    for i in range(len(label_name)):
        if count < train_size:
            if label_name[i] in class_name:
                labels.append(int(class_name[label_name[i]]))
                data[count] = raw_data[i]/255
                count+=1

    

    task_data = np.zeros((int(len(tasks_label)*each_task),32,32,1))
    task_label = []
    count = 0
    for i in range(len(data)):
        if count//each_task < len(tasks_label):
            key = count//each_task
            if labels[i] == tasks_label[key]:
                task_label.append(labels[i])
                task_data[count] = data[i]
                count+=1
    idx = np.arange(len(task_data))
    np.random.shuffle(idx)
    task_data = task_data[idx]
    task_label = np.array(task_label)[idx]
    
    return [task_data, np.array(task_label)]




def train_incremental(w_h, w_o,mem_set= [],task_1 =[], task_2 = [],num_task = NUM_TASKS, epochs =3):

    tasks = 2*[None]
    tasks[0] = task_1
    tasks[1] = task_2
    tasks_label = []
    tasks_label.append(list(set(list(task_1[1]))))
    tasks_label.append(list(set(list(task_2[1]))))
    for i in range(num_task):
        current_set = tasks[i]
        train_set = merge_set(mem_set, current_set) 
        for e in range(epochs):
            print("-" * 30)
            print(f"Training task {i} of epoch {e}\n")
            print("-" * 30)
            net = loihi2_net(dim =DIM,time_steps = TIME_STEPS,w_h = w_h,w_o=w_o)
            w_h, w_o=net.train_loihi_network([train_set[0][:150], train_set[1][:150]])
            del net
        '''
        Testing Past Dataset
        '''
        print(f"=================Testing Past Task:{i}=================")
        net = loihi2_net(dim =DIM,time_steps = TIME_STEPS,w_h = w_h,w_o=w_o)
        acc =net.test_loihi([mem_set[0][:100],mem_set[1][:100]])
        tank_name = get_tank_name(tasks_label[i])
        result = emstdp_results(testing_task= tank_name, training_sample=len(train_set[0]), testing_sample= 100, accuracy=acc)
        print(result)
        print(f"=================End Testing Past Task:{i}=================")
        print()
        new_mem_set = memory_data(current_set[0],current_set[1],tasks_label[i],k=50)
        mem_set = merge_set(mem_set,new_mem_set)

            
    return w_h, w_o
           


