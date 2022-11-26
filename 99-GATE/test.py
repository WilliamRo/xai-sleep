from typing import List
import numpy as np


configs = [f'test_data:{2*i},{2*i+1}' for i in range(10)]
a = np.arange(30).reshape(10,3)
b = np.zeros(15).reshape(5,3)
temp_index = 1 + np.arange(5, dtype=np.int)
random_channel = list(set([np.random.randint(0,3,dtype=np.int) for i in range(2)]))
print(a)
print('random channel:', random_channel)
for i in random_channel:
  a[list(temp_index), i] = b[:,i]
print(a)


