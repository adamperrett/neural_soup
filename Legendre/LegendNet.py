import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms, datasets
import copy
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import gc
import itertools
from tqdm import tqdm
from Legendre.train_mnist import NeuralNet

sns.set_theme(style="whitegrid")

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

batch_size = 64
trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, generator=torch.Generator(device=device))

# net_file = 'Legendre/mnist sigmoid hidden_size[200] test_acc[92.28].pt'
net_file = 'Legendre/mnist sigmoid hidden_size[20] test_acc[92.31].pt'
model = torch.load(net_file)

class VexSig(nn.Module):

    def __init__(self):
        super(VexSig, self).__init__()

    def forward(self, x, b, w):
        c = torch.square(w) * 0.05
        out = 1 / (1 + torch.exp(-(x * w) + b)) + (c * torch.square(x))
        return out

class VexDer(nn.Module):

    def __init__(self):
        super(VexDer, self).__init__()

    def forward(self, x, b, w):
        c = torch.square(w) * 0.05
        sig = 1 / (1 + torch.exp(-(x * w) + b))
        out = w * sig * (1 - sig) + (2 * c * x)
        return out

class CaveSig(nn.Module):

    def __init__(self):
        super(CaveSig, self).__init__()

    def forward(self, x, b, w):
        c = torch.square(w) * 0.05
        out = 1 / (1 + torch.exp(-(x * w) + b)) + (c * torch.square(x))
        return out

class CaveDer(nn.Module):

    def __init__(self):
        super(CaveDer, self).__init__()

    def forward(self, x, b, w):
        c = torch.square(w) * 0.05
        sig = 1 / (1 + torch.exp(-(x * w) + b))
        out = w * sig * (1 - sig) - (2 * c * x)
        return out


full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []
legendre_scale_factor = 10e3

net_values = [[] for i in range(len(g))]
net_total = []
vex_values = [[] for i in range(len(g))]
vex_total = []
cave_values = [[] for i in range(len(g))]
cave_total = []

for images, labels in train_loader:
    for weights, biases in zip(model.layer[0].weight.data, model.layer[0].bias.data):

        s_values[i].append(sigmoid(position, w, b))
        vex_values[i].append(vex_sigmoid(position, w, b))
        cave_values[i].append(cave_sigmoid(position, w, b))
        if not i:
            s_total.append(s_values[i][-1])
            full_vex_legendre_m.append(vex_der(position, w, b))
            vex_total.append(vex_values[i][-1])
            full_cave_legendre_m.append(cave_der(position, w, b))
            cave_total.append(cave_values[i][-1])
        else:
            s_total[-1] += s_values[i][-1]
            full_vex_legendre_m[-1] += vex_der(position, w, b)
            vex_total[-1] += vex_values[i][-1]
            full_cave_legendre_m[-1] += cave_der(position, w, b)
            cave_total[-1] += cave_values[i][-1]

    full_vex_legendre_c.append(vex_total[-1] - np.sum(full_vex_legendre_m[-1] * position))
    full_cave_legendre_c.append(cave_total[-1] - np.sum(full_cave_legendre_m[-1] * position))

print("Done")
