import numpy as np

def getData(size): 

    return np.random.gamma(2.0, 4.0, size = size).astype(np.float32)

