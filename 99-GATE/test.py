from typing import List
import numpy as np


a = np.stack([[1,2,3],[7,8,9],[4,5,6]], axis=-1)
a[:,1] = 0
print(a)

