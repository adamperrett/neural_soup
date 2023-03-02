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
model = torch.load('data/'+net_file+'.pt')

def Sig(x, w, b, out_w):
    out = out_w.unsqueeze(1) / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
    return torch.transpose(out, 0, 1)


def CaVexSig(x, w, out_w):
    # c = torch.square(w) * 0.5 * out_w
    c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
    out = torch.matmul(torch.square(x), c)
    return out


def Der(x, w, b, out_w, der_type):
    # c = torch.square(w) * 0.5 * out_w
    if der_type == 'sig':
        sig = 1 / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
        w_scale = torch.matmul(w.unsqueeze(1), out_w.unsqueeze(0))
        sig_derivative = (sig * (1 - sig))
        out = torch.stack([w_scale * der for der in sig_derivative])
    else:
        c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
        input_const = (2 * torch.stack([position.unsqueeze(1) * c for position in x]))
        out = input_const
    # full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
    return out

def piecewise_value(x, sig_m, sig_c, cavex_m, cavex_c, soft=True):
    sy = []
    cavexy = []
    for sm, sc, cvm, cvc in zip(sig_m, sig_c, cavex_m, cavex_c):
        smx = torch.matmul(x, sm)
        cavexx = torch.matmul(x, cvm)
        sy.append(smx + sc)
        cavexy.append(cavexx + cvc)
    sy = torch.stack(sy)
    cavexy = torch.stack(cavexy)
    if soft:
        soft_y = []
        for level in soft_levels:
            exp_cavex = torch.exp(cavexy / level)
            softness = exp_cavex / torch.sum(exp_cavex, dim=0)
            print(level, torch.min(softness), torch.max(softness))
            soft_y.append(torch.sum(sy * softness, dim=0))
        return soft_y
    else:
        max_vex_max = torch.max(cavexy, dim=0)[1]
        # min_cave_min = torch.min(-cavexy, dim=0)[1]
        vex_sy = torch.stack([torch.stack(
            [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max)])
        # cave_sy = torch.stack([torch.stack(
        #     [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min)])
        # y = (vex_sy + cave_sy) / 2
        y = vex_sy
        return [y]

# full_sig = []
# full_cavex = []
# full_sig_der = []
# full_cavex_der = []
# full_sig_c = []
# full_cavex_c = []
#
# legendre_scale_factor = 1#0e3
#
# hidden_size = model.layer[0].bias.shape[0]
#
#
# print("extracting Legendre transform")
# for images, labels in tqdm(train_loader):
#     images = images.reshape(-1, 784).to(torch.device(device))
#     for i, (w, b, out_w) in enumerate(zip(model.layer[0].weight.data,
#                                    model.layer[0].bias.data,
#                                    torch.transpose(model.layer[1].weight.data, 0, 1)
#                                    )):
#
#         neuron_sig = Sig(images, w, b, out_w)
#         neuron_cavex = CaVexSig(images, w, out_w)
#         neuron_sig_der = Der(images, w, b, out_w, 'sig')
#         neuron_cavex_der = Der(images, w, b, out_w, False)
#
#         for output, ow in enumerate(out_w):
#             if ow < 0:
#                 neuron_cavex[:, output] *= -1
#                 neuron_cavex_der[:, :, output] *= -1
#
#         if not i:
#             full_sig.append(neuron_sig)
#             full_cavex.append(neuron_cavex)
#             full_sig_der.append(neuron_sig_der)
#             full_cavex_der.append(neuron_cavex_der)
#         else:
#             full_sig[-1] += neuron_sig
#             full_cavex[-1] += neuron_cavex
#             full_sig_der[-1] += neuron_sig_der
#             full_cavex_der[-1] += neuron_cavex_der
#
#     full_sig_c.append(full_sig[-1] - torch.stack(
#         [torch.matmul(im, f) for f, im in zip(full_sig_der[-1], images)]))
#     full_cavex_c.append(full_cavex[-1] - torch.stack(
#         [torch.matmul(im, f) for f, im in zip(full_cavex_der[-1], images)]))
#
# full_sig_c = torch.vstack(full_sig_c)
# full_cavex_c = torch.vstack(full_cavex_c)
# full_sig_der = torch.vstack(full_sig_der)
# full_cavex_der = torch.vstack(full_cavex_der)
#
# torch.save(full_sig_c, 'data/sig_c {}.pt'.format(net_file))
# torch.save(full_cavex_c, 'data/cavex_c {}.pt'.format(net_file))
# torch.save(full_sig_der, 'data/sig_der {}.pt'.format(net_file))
# torch.save(full_cavex_der, 'data/cavex_der {}.pt'.format(net_file))

full_sig_c = torch.load('data/sig_c {}.pt'.format(net_file))
full_cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))
full_sig_der = torch.load('data/sig_der {}.pt'.format(net_file))
full_cavex_der = torch.load('data/cavex_der {}.pt'.format(net_file))

soft_levels = [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# with torch.no_grad():
#     correct_m = 0
#     correct_l = 0
#     total = 0
#     for images, labels in tqdm(train_loader):
#         images = images.reshape(-1, 784).to(torch.device(device))
#         out_m = model(images)
#         _, pred = torch.max(out_m, 1)
#         correct_m += (pred == labels).sum().item()
#
#         out_l = piecewise_value(images, full_sig_der, full_sig_c, full_cavex_der, full_cavex_c)
#
#         _, pred = torch.max(out_l, 1)
#         correct_l += (pred == labels).sum().item()
#
#         total += labels.size(0)
#         print("Current total", total+1)
#         print('Model training accuracy: {} %'.format(100 * correct_m / total))
#         print('Legendre training accuracy: {} %'.format(100 * correct_l / total))
#     training_accuracies = [100 * np.array(correct_m) / total, 100 * np.array(correct_l) / total]

with torch.no_grad():
    correct_m = 0
    correct_l = [0 for i in range(len(soft_levels))]
    total = 0
    for images, labels in tqdm(train_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

        out_m = model(images)
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_l = piecewise_value(images, full_sig_der, full_sig_c, full_cavex_der, full_cavex_c)

        for i, out in enumerate(out_l):
            _, pred = torch.max(out, 1)
            correct_l[i] += (pred == labels).sum().item()

        total += labels.size(0)
        print("Current total", total)
        print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Legendre testing')
        for i in range(len(soft_levels)):
            print("Accuracy:", 100 * correct_l[i] / total, " Softness:", soft_levels[i])
    testing_accuracies = [100 * np.array(correct_m) / total, 100 * np.array(correct_l) / total]

# print("Training accuracies:", training_accuracies)
print("Testing accuracies:", testing_accuracies)


print("Done")
