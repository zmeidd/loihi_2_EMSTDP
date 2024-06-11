import numpy as np
from ood import detector
from utils import pre_process_data
data = np.load("./files/min_data.npy")[:20]
label = np.load("./files/min_label.npy")[:20]
data = pre_process_data(data)
result = detector(data)
print(label)
print(result)
