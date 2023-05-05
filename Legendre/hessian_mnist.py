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
import sys
sys.path.append('/mnt/iusers01/gb01/mbaxrap7/scratch/neural_soup/Legendre')
sys.path.append('/mnt/iusers01/gb01/mbaxrap7/scratch/neural_soup')
from Legendre.train_cnn import CNN
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
import sys
file_number = int(sys.argv[1])
# net_file = 'mnist sigmoid hidden_size[2000] test_acc[98.1]'
# net_file = 'mnist sigmoid hidden_size[200] test_acc[98.05]'
# net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'
# net_file = 'mnist0.5 sigmoid hidden_size[200, 200] test_acc[98.11]'
# net_file = 'mnist0.5 relu hidden_size[200, 200] test_acc[98.51]'
# net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[200, 200] test_acc[98.11]'

if file_number == 1:
    net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[200, 200, 200, 200, 200] test_acc[98.47]'
if file_number == 2:
    net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[200, 200, 200, 200] test_acc[98.53]'
if file_number == 3:
    net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[400, 400, 400, 400] test_acc[98.54]'

# if file_number == 1:
#     net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[200, 200, 200, 200] test_acc[97.49]'
# if file_number == 2:
#     net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[200, 200, 200] test_acc[97.9]'
# if file_number == 3:
#     net_file = 'mnist0.5 sigmoid hidden_size[200, 200] test_acc[98.11]'
# if file_number == 4:
#     net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[1600] test_acc[98.27]'
# if file_number == 5:
#     net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[800] test_acc[98.18]'
# if file_number == 6:
#     net_file = 'mnist0.5 sigmoid cnnFalse hidden_size[400] test_acc[98.24]'
# if file_number == 7:
#     net_file = 'mnist0.5 sigmoid cnnTrue hidden_size[200] test_acc[99.04]'
# if file_number == 8:
#     net_file = 'mnist0.5 sigmoid hidden_size[200] test_acc[98.1]'
# if file_number == 9:
#     net_file = 'mnist0.5 relu cnnFalse hidden_size[200, 200] test_acc[98.5]'


conv = False
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

    margin_of_error = 0.#0001
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
    max_vex_max3 = torch.max(cavexy, dim=2)
    min_cave_min3 = torch.min(-cavexy, dim=2)
    vex_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3[1])])
    cave_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3[1])])
    # if max_mindex:
    #     y = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    # else:
    #     y = torch.transpose((y_min + y_max) / 2, 0, 1)

    vex_cave = torch.transpose((y_min + y_max) / 2, 0, 1)
    indexing = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    vex_sub_vex = torch.transpose((y_max - max_vex_max3[0]), 0, 1)
    vex_add_cave = torch.transpose((y_max + min_cave_min3[0]), 0, 1)
    cave_add_vex = torch.transpose((y_min + max_vex_max3[0]), 0, 1)
    cave_sub_cave = torch.transpose((y_min - min_cave_min3[0]), 0, 1)
    y_max_cavexy = torch.stack([torch.stack(
        [cavexy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(torch.max(vexy, dim=2)[1])])
    sub_indexed = torch.transpose((y_max - y_max_cavexy), 0, 1)
    cavexy_y_max = torch.stack([torch.stack(
        [vexy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3[1])])
    indexed_y_sub_vex = torch.transpose((cavexy_y_max - max_vex_max3[0]), 0, 1)

    return vex_cave, indexing, vex_sub_vex, vex_add_cave, cave_add_vex, cave_sub_cave, sub_indexed, indexed_y_sub_vex
    # return torch.transpose((y_min + y_max) / 2, 0, 1), torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
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

# hidden_size = model.layer[0].bias.shape[0]


print("extracting Legendre transform")

with torch.no_grad():
    print("calculating Legendre constant")
    # c_list = []
    # for images, labels in tqdm(c_loader):
    #     images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
    #     cavex_const = calculate_c(model, images)
    #     c_list.append(cavex_const)
    # cavex_const = torch.max(torch.stack(c_list), dim=0)[0]
    # cavex_const = torch.ones(num_outputs) * 7
    cavex_const = torch.tensor(
        [0.5253, 0.4542, 0.6770, 0.6772, 0.9017, 0.8467, 0.7145, 0.6749, 1.0000, 0.8234]) * 9
    # cavex_const = torch.tensor([
    #     5.1569, 4.4589, 6.6459, 6.6473, 8.8517, 8.3111, 7.0140, 6.6251, 9.8162, 8.0831])
    # tensor([0.5253, 0.4542, 0.6770, 0.6772, 0.9017, 0.8467, 0.7145, 0.6749, 1.0000,
    #         0.8234]) # collected from cavex_c of sigmoid clustered planes

    print(cavex_const)
    # 21
    # tensor([5.8621, 4.7964, 8.7576, 7.8061, 9.5018, 9.4767, 8.2469, 7.4933,
    #         11.0814, 9.2439])
    # tensor([5.1569, 4.4589, 6.6459, 6.6473, 8.8517, 8.3111, 7.0140, 6.6251, 9.8162,
    #         8.0831])
    # tensor([0.3176, 0.2004, 0.5969, 0.4225, 0.5185, 0.6505, 0.4972, 0.4749, 0.7302,
    #         0.4722])
    # tensor([6.1096, 5.0601, 8.4366, 7.9147, 10.4071, 10.2626, 8.5056, 8.0497,
    #         12.0068, 9.4996])

print("calculating Legendre planes")
for images, labels in tqdm(train_loader):
    if conv:
        images = images.to(torch.device(device)) - 0.5
    else:
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

    images.requires_grad = True
    n_out = []
    n_m = []
    for out in range(num_outputs):
        n_out.append(model.separate_outputs(out)(images))
        n_out[-1].backward(torch.ones(images.shape[0]))
        n_m.append(images.grad.clone().detach().reshape(-1, 784))
        images.grad = None
        n_out[-1].detach()
    images.requires_grad = False

    if conv:
        images = images.reshape(-1, 784)
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
cavex_out = torch.hstack(cavex_out)
cavex_m = torch.hstack(cavex_m)
cavex_c = torch.hstack(cavex_c)

print('', end='')

torch.save(net_m, 'data/net_m {}.pt'.format(net_file))
torch.save(net_c, 'data/net_c {}.pt'.format(net_file))
torch.save(cavex_m, 'data/cavex_m {}.pt'.format(net_file))
torch.save(cavex_c, 'data/cavex_c {}.pt'.format(net_file))

# net_m = torch.load('data/net_m {}.pt'.format(net_file))
# net_c = torch.load('data/net_c {}.pt'.format(net_file))
# cavex_m = torch.load('data/cavex_m {}.pt'.format(net_file))
# cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))


print("calculating testing accuracy")
metric_name = ['vex_cave', 'indexing', 'vex_sub_vex', 'vex_add_cave', 'cave_add_vex', 'cave_sub_cave', 'sub_indexed', 'indexed_y_sub_vex']
metrics = len(metric_name)
with torch.no_grad():
    correct_m = 0
    correct_out = [0 for i in range(metrics)]
    total = 0
    all_m = []
    all_out = [[] for i in range(metrics)]
    all_mindex = []
    for images, labels in tqdm(test_loader):
        if conv:
            images = images.to(torch.device(device)) - 0.5
        else:
            images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        all_m.append(out_m)

        images = images.reshape(-1, 784)
        various_out = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)

        for out, out_v in enumerate(various_out):
            _, pred = torch.max(out_v, 1)
            correct_out[out] += (pred == labels).sum().item()
            all_out[out].append(out_v)

        total += labels.size(0)

print("Current total {}".format(total))
print('Model testing accuracy: {} %'.format(100 * correct_m / total))
for correct, name in zip(correct_out, metric_name):
    print('{} testing accuracy: {} %'.format(name, 100 * correct / total))
all_m = torch.vstack(all_m)
all_out = [torch.vstack(out) for out in all_out]
for out, name in zip(all_out, metric_name):
    print(name, "difference", torch.sum(torch.abs(all_m - out)))

print("calculating training accuracy")
with torch.no_grad():
    correct_m = 0
    correct_out = [0 for i in range(metrics)]
    total = 0
    all_m = []
    all_out = [[] for i in range(metrics)]
    all_mindex = []
    for images, labels in tqdm(train_loader):
        if conv:
            images = images.to(torch.device(device)) - 0.5
        else:
            images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        all_m.append(out_m)

        images = images.reshape(-1, 784)
        various_out = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)

        for out, out_v in enumerate(various_out):
            _, pred = torch.max(out_v, 1)
            correct_out[out] += (pred == labels).sum().item()
            all_out[out].append(out_v)

        total += labels.size(0)

print("Current total {}".format(total))
print('Model training accuracy: {} %'.format(100 * correct_m / total))
for correct, name in zip(correct_out, metric_name):
    print('{} training accuracy: {} %'.format(name, 100 * correct / total))
all_m = torch.vstack(all_m)
all_out = [torch.vstack(out) for out in all_out]
for out, name in zip(all_out, metric_name):
    print(name, "difference", torch.sum(torch.abs(all_m - out)))

print("Done")
