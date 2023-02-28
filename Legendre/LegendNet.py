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
from Legendre.train_2D import NeuralNet
from Legendre.train_2D import generate_corner_2class_data, generate_xor_data
sns.set_theme(style="whitegrid")

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

torch.manual_seed(272727)
print("generating data")
batch_size = 128
trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, generator=torch.Generator(device=device))

print("loading net")
# net_file = 'mnist sigmoid hidden_size[2000] test_acc[98.1]'
# net_file = 'mnist sigmoid hidden_size[200] test_acc[98.05]'
net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'
model = torch.load(net_file+'.pt')

def Sig(x, w, b, out_w):
    out = out_w.unsqueeze(1) / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
    return torch.transpose(out, 0, 1)


def CaVexSig(x, w, b, out_w, vex):
    c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
    if vex:
        out = Sig(x, w, b, out_w) + torch.matmul(torch.square(x), c)
    else:
        out = Sig(x, w, b, out_w) - torch.matmul(torch.square(x), c)
    return out


def CaVexDer(x, w, b, out_w, vex):
    c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
    sig = 1 / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
    w_scale = torch.matmul(w.unsqueeze(1), out_w.unsqueeze(0))
    input_const = (2 * torch.stack([position.unsqueeze(1) * c for position in x]))
    sig_derivative = (sig * (1 - sig))
    if vex:
        out = torch.stack([w_scale * der for der in sig_derivative]) + input_const
    else:
        out = torch.stack([w_scale * der for der in sig_derivative]) - input_const
    return out

def piecewise_value(x, legendre_m, legendre_c, vex=True, soft=False):
    y = []
    for m, c in zip(legendre_m, legendre_c):
        mx = torch.matmul(x, m)
        y.append((mx + c) / legendre_scale_factor)
    y = torch.stack(y)
    if soft:
        temperature = .01
        if vex:
            return torch.sum(y * torch.exp(y / temperature) / torch.sum(torch.exp(y / temperature)))
        else:
            return torch.sum(y * torch.exp(-y / temperature) / torch.sum(torch.exp(-y / temperature)))
    else:
        if vex:
            return torch.max(y, dim=0)[0]
        else:
            return torch.min(y, dim=0)[0]


legendre_scale_factor = 1#0e3
full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []

hidden_size = model.layer[0].bias.shape[0]
# net_values = [[] for i in range(hidden_size)]
# net_total = []
# vex_values = [[] for i in range(hidden_size)]
vex_total = []
# cave_values = [[] for i in range(hidden_size)]
cave_total = []

print("extracting Legendre transform")
for images, labels in tqdm(train_loader):
    images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
    for i, (w, b, out_w) in enumerate(zip(model.layer[0].weight.data,
                                   model.layer[0].bias.data,
                                   torch.transpose(model.layer[1].weight.data, 0, 1)
                                   )):

        # net_values[i].append(Sig(images, w, b, out_w))
        # vex_values[i].append(CaVexSig(images, w, b, out_w, True))
        # cave_values[i].append(CaVexSig(images, w, b, out_w, False))
        neuron_vex_total = CaVexSig(images, w, b, out_w, True)
        neuron_cave_total = CaVexSig(images, w, b, out_w, False)
        neuron_vex_legendre_m = CaVexDer(images, w, b, out_w, True)
        neuron_cave_legendre_m = CaVexDer(images, w, b, out_w, False)

        for output, ow in enumerate(out_w):
            if ow < 0:
                temp = neuron_vex_legendre_m[:, :, output].clone().detach()
                neuron_vex_legendre_m[:, :, output] = neuron_cave_legendre_m[:, :, output]
                neuron_cave_legendre_m[:, :, output] = temp
                temp = neuron_cave_total[:, output].clone().detach()
                neuron_cave_total[:, output] = neuron_vex_total[:, output]
                neuron_vex_total[:, output] = temp

        if not i:
            # net_total.append(net_values[i][-1])
            full_vex_legendre_m.append(neuron_vex_legendre_m)
            vex_total.append(neuron_vex_total)
            full_cave_legendre_m.append(neuron_cave_legendre_m)
            cave_total.append(neuron_cave_total)
        else:
            # net_total[-1] += net_values[i][-1]
            full_vex_legendre_m[-1] += neuron_vex_legendre_m
            vex_total[-1] += neuron_vex_total
            full_cave_legendre_m[-1] += neuron_cave_legendre_m
            cave_total[-1] += neuron_cave_total

    # real_net = model(images)

    full_vex_legendre_c.append(vex_total[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(full_vex_legendre_m[-1], images)]))
    full_cave_legendre_c.append(cave_total[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(full_cave_legendre_m[-1], images)]))

torch.cuda.empty_cache()
full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
full_cave_legendre_c = torch.vstack(full_cave_legendre_c)

torch.save(full_vex_legendre_m, 'vex_m {}.pt'.format(net_file))
torch.save(full_vex_legendre_c, 'vex_c {}.pt'.format(net_file))
torch.save(full_cave_legendre_m, 'cave_m {}.pt'.format(net_file))
torch.save(full_cave_legendre_c, 'cave_c {}.pt'.format(net_file))

# full_vex_legendre_m = torch.load('vex_m.pt')
# full_vex_legendre_c = torch.load('vex_c.pt')
# full_cave_legendre_m = torch.load('cave_m.pt')
# full_cave_legendre_c = torch.load('cave_c.pt')

all_vex_output = []
all_cave_output = []
with torch.no_grad():
    correct_m = 0
    correct_l = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        out_m = model(images)
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        vex = piecewise_value(images, full_vex_legendre_m, full_vex_legendre_c)
        cave = piecewise_value(images, full_vex_legendre_m, full_vex_legendre_c, vex=False)
        out_l = (vex + cave) * legendre_scale_factor / 2
        _, pred = torch.max(out_l, 1)
        correct_l += (pred == labels).sum().item()

        all_vex_output.append(vex)
        all_cave_output.append(cave)

        total += labels.size(0)
        print("Current total {}/{}".format(total+1, len(full_vex_legendre_m)))
        print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
    # testing_accuracies = 100 * np.array(correct) / total


print("Done")
