import numpy as np
from ood import detector
data = np.load("out_data.npy")[:20]
result = detector(data)
print(result)
