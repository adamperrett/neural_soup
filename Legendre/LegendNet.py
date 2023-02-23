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

# net_file = 'mnist sigmoid hidden_size[200] test_acc[92.28].pt'
net_file = 'mnist sigmoid hidden_size[20] test_acc[92.31].pt'
model = torch.load(net_file)

def Sig(x, w, b):
        out = 1 / (1 + torch.exp(-(x * w) + b))
        return out

def VexSig(x, w, b):
        c = torch.square(w) * 0.05
        out = 1 / (1 + torch.exp(-(x * w) + b)) + (c * torch.square(x))
        return out

def VexDer(x, w, b):
        c = torch.square(w) * 0.05
        sig = 1 / (1 + torch.exp(-(x * w) + b))
        out = w * sig * (1 - sig) + (2 * c * x)
        return out

def CaveSig(x, w, b):
        c = torch.square(w) * 0.05
        out = 1 / (1 + torch.exp(-(x * w) + b)) + (c * torch.square(x))
        return out

def CaveDer(x, w, b):
        c = torch.square(w) * 0.05
        sig = 1 / (1 + torch.exp(-(x * w) + b))
        out = w * sig * (1 - sig) - (2 * c * x)
        return out


full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []
legendre_scale_factor = 10e3

hidden_size = model.layer[0].bias.shape[0]
net_values = [[] for i in range(hidden_size)]
net_total = []
vex_values = [[] for i in range(hidden_size)]
vex_total = []
cave_values = [[] for i in range(hidden_size)]
cave_total = []

for images, labels in train_loader:
    images = images.reshape(-1, 784).to(torch.device(device))
    for i, (w, b, out_w, out_b) in enumerate(zip(model.layer[0].weight.data,
                                   model.layer[0].bias.data,
                                   model.layer[1].weight.data,
                                   model.layer[1].bias.data
                                   )):
        net_values[i].append(Sig(images, w, b))
        vex_values[i].append(VexSig(images, w, b))
        cave_values[i].append(CaveSig(images, w, b))
        if not i:
            net_total.append(net_values[i][-1])
            full_vex_legendre_m.append(VexDer(images, w, b))
            vex_total.append(vex_values[i][-1])
            full_cave_legendre_m.append(CaveDer(images, w, b))
            cave_total.append(cave_values[i][-1])
        else:
            net_total[-1] += net_values[i][-1]
            full_vex_legendre_m[-1] += VexDer(images, w, b)
            vex_total[-1] += vex_values[i][-1]
            full_cave_legendre_m[-1] += CaveDer(images, w, b)
            cave_total[-1] += cave_values[i][-1]

    full_vex_legendre_c.append(vex_total[-1] - torch.sum(full_vex_legendre_m[-1] * images))
    full_cave_legendre_c.append(cave_total[-1] - torch.sum(full_cave_legendre_m[-1] * images))

full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
full_cave_legendre_c = torch.vstack(full_cave_legendre_c)

print("Done")
