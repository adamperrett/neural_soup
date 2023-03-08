import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms, datasets
from torch.profiler import profile, record_function, ProfilerActivity
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

default_type = torch.float32#64
torch.set_default_dtype(default_type)

torch.manual_seed(272727)
print("generating data")
batch_size = 128
c_size = 128
trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
c_loader = torch.utils.data.DataLoader(trainset, batch_size=c_size,
                                           shuffle=True, generator=torch.Generator(device=device))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, generator=torch.Generator(device=device))

print("loading net")
# net_file = 'mnist sigmoid hidden_size[2000] test_acc[98.1]'
# net_file = 'mnist sigmoid hidden_size[200] test_acc[98.05]'
net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'
model = torch.load('data/'+net_file+'.pt')

num_outputs = 10

def calculate_c(model, all_x):
    net_hessian = torch.stack([torch.stack([
        torch.autograd.functional.hessian(model.separate_outputs(out), (all_x[i:i + 1])).squeeze()
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    eigen_values = torch.stack([torch.stack([
        torch.linalg.eig(net_hessian[i][out])[0].real
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    margin_of_error = 0.0001
    cavex_c = torch.stack([torch.stack([
        torch.max(
            -torch.min(torch.hstack([
                eigen_values[i][out], torch.tensor(0)]))) + margin_of_error
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    output_c = torch.max(cavex_c, dim=0)

    return output_c[0]

def piecewise_value(x, net_m, net_c, cavex_m, cavex_c, max_mindex=False):
    # if max_mindex:
    #     sy = []
    #     cavexy = []
    #     for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
    #         smx = torch.matmul(x, sm)
    #         cavexx = torch.matmul(x, cvm)
    #         sy.append(smx + sc)
    #         cavexy.append(cavexx + cvc)
    #     sy = torch.stack(sy)
    #     cavexy = torch.stack(cavexy)
    #     max_vex_max3 = torch.max(cavexy, dim=0)[1]
    #     # min_cave_min3 = torch.min(-cavexy, dim=0)[1]
    #     vex_sy = torch.stack([torch.stack(
    #         [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3)])
    #     # cave_sy = torch.stack([torch.stack(
    #     #     [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
    #     # y = (vex_sy + cave_sy) / 2
    #     y = vex_sy
    # else:
    #     vexy = []
    #     cavey = []
    #     for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
    #         vexx = torch.matmul(x, sm + cvm)
    #         cavex = torch.matmul(x, sm - cvm)
    #         vexy.append(vexx + cvc + sc)
    #         cavey.append(cavex - cvc + sc)
    #     vexy = torch.stack(vexy)
    #     cavey = torch.stack(cavey)
    #     y_max = torch.max(vexy, dim=0)[0]
    #     y_min = torch.min(cavey, dim=0)[0]
    #     y = (y_min + y_max) / 2
    # return y
    sy = []
    vexy = []
    cavey = []
    cavexy = []
    for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
        smx = torch.matmul(x, torch.transpose(sm, 0, 1))
        vexx = torch.matmul(x, torch.transpose(sm + cvm, 0, 1))
        cavex = torch.matmul(x, torch.transpose(sm - cvm, 0, 1))
        cavexx = torch.matmul(x, torch.transpose(cvm, 0, 1))
        sy.append((smx + sc))
        vexy.append((vexx + cvc + sc))
        cavey.append((cavex - cvc + sc))
        cavexy.append(cavexx + cvc)
    sy = torch.stack(sy)
    vexy = torch.stack(vexy)
    cavey = torch.stack(cavey)
    cavexy = torch.stack(cavexy)
    y_max = torch.max(vexy, dim=2)[0]
    y_min = torch.min(cavey, dim=2)[0]
    # max_vex_max = torch.max(vexy, dim=2)[1]
    # min_cave_min = torch.min(cavey, dim=2)[1]
    # max_vex_max2 = torch.max(sy + cavexy, dim=2)[1]
    # min_cave_min2 = torch.min(sy - cavexy, dim=2)[1]
    max_vex_max3 = torch.max(cavexy, dim=2)[1]
    min_cave_min3 = torch.min(-cavexy, dim=2)[1]
    vex_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3)])
    cave_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
    # if max_mindex:
    #     y = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    # else:
    #     y = torch.transpose((y_min + y_max) / 2, 0, 1)
    return torch.transpose((y_min + y_max) / 2, 0, 1), torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    # if soft:
    #     temperature = .01
        # if vex:
        #     return torch.sum(y * torch.exp(y / temperature) / torch.sum(torch.exp(y / temperature)))
        # else:
        #     return torch.sum(y * torch.exp(-y / temperature) / torch.sum(torch.exp(-y / temperature)))
    # else:
    #     if vex:
    #         return torch.max(y, dim=0)[0]
    #     else:
    #         return torch.min(y, dim=0)[0]

net_out = []
net_m = []
net_c = []
cavex_out = []
cavex_m = []
cavex_c = []

hidden_size = model.layer[0].bias.shape[0]


print("extracting Legendre transform")

with torch.no_grad():
    print("calculating Legendre constant")
    c_list = []
    for images, labels in tqdm(c_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        cavex_const = calculate_c(model, images)
        c_list.append(cavex_const)
    cavex_const = torch.max(torch.stack(c_list), dim=0)[0]
    # cavex_const = torch.ones(num_outputs) * 3
    print(cavex_const)

print("calculating Legendre planes")
for images, labels in tqdm(train_loader):
    images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

    images.requires_grad = True
    n_out = []
    n_m = []
    for out in range(num_outputs):
        n_out.append(model.separate_outputs(out)(images))
        n_out[-1].backward(torch.ones(images.shape[0]))
        n_m.append(images.grad.clone().detach())
        images.grad = None
        n_out[-1].detach()
    images.requires_grad = False

    with torch.no_grad():
        x2_out = torch.sum(images * images, dim=1)
        cx2 = 0.5 * torch.stack([torch.sum(images * images, dim=1) * c for c in cavex_const])
        cx2_grad = torch.stack([images * c for c in cavex_const])

        net_out.append(torch.stack(n_out))
        net_m.append(torch.stack(n_m))
        net_c.append(torch.stack([
            n_out[out] - torch.sum(n_m[out] * images, dim=1) for out in range(num_outputs)
        ]))
        cavex_out.append(cx2)
        cavex_m.append(cx2_grad)
        cavex_c.append(cx2 - torch.sum(cx2_grad * images, dim=2))

# net_out = torch.hstack(net_out)
net_m = torch.hstack(net_m)
net_c = torch.hstack(net_c)
# cavex_out = torch.hstack(cavex_out)
cavex_m = torch.hstack(cavex_m)
cavex_c = torch.hstack(cavex_c)

print('', end='')

# full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
# full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
# full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
# full_cave_legendre_c = torch.vstack(full_cave_legendre_c)
# full_vex_legendre_m = torch.load('vex_m.pt')
# full_vex_legendre_c = torch.load('vex_c.pt')
# full_cave_legendre_m = torch.load('cave_m.pt')
# full_cave_legendre_c = torch.load('cave_c.pt')

print("calculating testing accuracy")
with torch.no_grad():
    correct_m = 0
    correct_l = 0
    correct_mindex = 0
    total = 0
    all_m = []
    all_l = []
    all_mindex = []
    for images, labels in tqdm(test_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_l, mindex_out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)
        _, pred = torch.max(out_l, 1)
        correct_l += (pred == labels).sum().item()

        _, pred = torch.max(mindex_out_l, 1)
        correct_mindex += (pred == labels).sum().item()

        total += labels.size(0)

        all_m.append(out_m)
        all_l.append(out_l)
        all_mindex.append(mindex_out_l)

print("Current total {}".format(total))
print('Model testing accuracy: {} %'.format(100 * correct_m / total))
print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
print('Mindex Legendre testing accuracy: {} %'.format(100 * correct_mindex / total))
all_m = torch.vstack(all_m)
all_l = torch.vstack(all_l)
all_mindex = torch.vstack(all_mindex)
print("Legendre difference", torch.sum(torch.abs(all_m - all_l)))
print("Minmax Legendre difference", torch.sum(torch.abs(all_m - all_mindex)))

print("calculating training accuracy")
with torch.no_grad():
    correct_m = 0
    correct_l = 0
    correct_mindex = 0
    total = 0
    all_m = []
    all_l = []
    all_mindex = []
    for images, labels in tqdm(train_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_l, mindex_out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)
        _, pred = torch.max(out_l, 1)
        correct_l += (pred == labels).sum().item()

        _, pred = torch.max(mindex_out_l, 1)
        correct_mindex += (pred == labels).sum().item()

        total += labels.size(0)

        all_m.append(out_m)
        all_l.append(out_l)
        all_mindex.append(mindex_out_l)

print("Current total {}".format(total))
print('Model testing accuracy: {} %'.format(100 * correct_m / total))
print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
print('Mindex Legendre testing accuracy: {} %'.format(100 * correct_mindex / total))
all_m = torch.vstack(all_m)
all_l = torch.vstack(all_l)
all_mindex = torch.vstack(all_mindex)
print("Legendre difference", torch.sum(torch.abs(all_m - all_l)))
print("Minmax Legendre difference", torch.sum(torch.abs(all_m - all_mindex)))

print("Done")
