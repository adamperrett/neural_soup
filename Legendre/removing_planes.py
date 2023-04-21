import time
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"



'''
removal techniques:
    iteratively remove most similar
    batch remove similar?
    different distance metrics
    remove separately
'''

net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'

print("loading files")
net_m = torch.load('data/net_m {}.pt'.format(net_file))
net_c = torch.load('data/net_c {}.pt'.format(net_file))
cavex_m = torch.load('data/cavex_m {}.pt'.format(net_file))
cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))

cave_m = net_m - cavex_m
cave_c = net_c - cavex_c
vex_m = net_m + cavex_m
vex_c = net_c + cavex_c

# full_cavex = torch.hstack([cavex_m.reshape([cavex_m.shape[0]*cavex_m.shape[1], cavex_m.shape[2]]),
#                            cavex_c.reshape([cavex_c.shape[0]*cavex_c.shape[1], 1])])
# full_net = torch.hstack([net_m.reshape([net_m.shape[0]*net_m.shape[1], net_m.shape[2]]),
#                          net_c.reshape([net_c.shape[0]*net_c.shape[1], 1])])
# print("Stacking")
# full_all = torch.hstack([net_m.reshape([net_m.shape[0]*net_m.shape[1], net_m.shape[2]]),
#                          net_c.reshape([net_c.shape[0]*net_c.shape[1], 1]),
#                          cavex_m.reshape([cavex_m.shape[0] * cavex_m.shape[1], cavex_m.shape[2]]),
#                          cavex_c.reshape([cavex_c.shape[0] * cavex_c.shape[1], 1])
#                          ])
# # full_all = torch.hstack([full_cavex, full_net])
# torch.save(full_all, 'data/full_all {}.pt'.format(net_file))
# full_all = torch.load('data/full_all {}.pt'.format(net_file))



# torch.save(net_m, 'data/net_m {}.pt'.format(net_file))
# torch.save(net_c, 'data/net_c {}.pt'.format(net_file))
# torch.save(cavex_m, 'data/cavex_m {}.pt'.format(net_file))
# torch.save(cavex_c, 'data/cavex_c {}.pt'.format(net_file))



print("Done")
