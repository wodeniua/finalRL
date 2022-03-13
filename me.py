import random
from env_config import *
import numpy as np
n_random_fixed = 2
BIN_MAX_COUNT = 10
total_items = []
# for i in range(n_random_fixed):
#     bins_list = []
#     # for i, b in enumerate(range(BIN_MAX_COUNT)):
#     for i in range(BIN_MAX_COUNT):
#         x = random.randint(2, 5)
#         y = random.randint(2, 5)
#         w = random.randint(2, 5)
#         bins_list.append((x, y, w))
#     total_items.append(bins_list)
#     print(bins_list)
# print(total_items)
# print(len(bins_list),len(total_items))
# p = np.zeros((3, 4, 5))
# print(p.size)
# print(p)
p = np.zeros((ENV.ROW_COUNT, ENV.COL_COUNT, 3))
p[:,:,2] = -1
# p = np.full((ENV.ROW_COUNT, ENV.COL_COUNT, 3),-1)
print(p)

