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

default_type = torch.float32#64
torch.set_default_dtype(default_type)

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

# def VexSig(x, w, b, out_w):
#     # should c also be 2D?
#     # is the sum over the right dimension
#     # c = torch.square(w) * 0.5 * out_w
#     # c = torch.matmul(torch.square(w) * 0.5, out_w)
#     c = torch.matmul(torch.square(w).unsqueeze(1) * 0.5, out_w.unsqueeze(0))
#     out = Sig(x, w, b, out_w) + torch.matmul(torch.square(x), c)
#     return out
#
# def VexDer(x, w, b, out_w):
#     # c = torch.square(w) * 0.5 * out_w
#     c = torch.matmul(torch.square(w).unsqueeze(1) * 0.5, out_w.unsqueeze(0))
#     sig = 1 / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
#     w_scale = torch.matmul(w.unsqueeze(1), out_w.unsqueeze(0))
#     input_const = (2 * torch.stack([position.unsqueeze(1) * torch.transpose(c, 0, 1) for position in x]))
#     sig_derivative = (sig * (1 - sig))
#     out = torch.stack([w_scale*der for der in sig_derivative]) + input_const
#     # full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
#     return out
#
# def CaveSig(x, w, b, out_w):
#     # c = torch.square(w) * 0.5 * out_w
#     c = torch.matmul(torch.square(w).unsqueeze(1) * 0.5, out_w.unsqueeze(0))
#     out = Sig(x, w, b, out_w) - torch.matmul(torch.square(x), c)
#     return out
#
# def CaveDer(x, w, b, out_w):
#     # c = torch.square(w) * 0.5 * out_w
#     c = torch.matmul(torch.square(w).unsqueeze(1) * 0.5, out_w.unsqueeze(0))
#     sig = 1 / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
#     w_scale = torch.matmul(w.unsqueeze(1), out_w.unsqueeze(0))
#     input_const = (2 * torch.stack([position.unsqueeze(1) * c for position in x]))
#     sig_derivative = (sig * (1 - sig))
#     out = torch.stack([w_scale * der for der in sig_derivative]) - input_const
#     # full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
#     return out

def CaVexSig(x, w, out_w, old=False):
    if old:
        c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
        out = torch.matmul(torch.square(x), c)
    else:
        c = (torch.matmul(w.unsqueeze(0), w.unsqueeze(1)) * out_w) * 0.05
        x_squared = torch.matmul(torch.transpose(x.unsqueeze(2), 1, 2), x.unsqueeze(2)).squeeze()
        out = x_squared.unsqueeze(1) * c
    return out


def Der(x, w, b, out_w, der_type, old=False):
    # c = torch.square(w) * 0.5 * out_w
    if der_type == 'sig':
        sig = 1 / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
        w_scale = torch.matmul(w.unsqueeze(1), out_w.unsqueeze(0))
        sig_derivative = (sig * (1 - sig))
        out = torch.stack([w_scale * der for der in sig_derivative])
    else:
        if old:
            c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
        else:
            c = (torch.matmul(w.unsqueeze(0), w.unsqueeze(1)) * out_w) * 0.05
        input_const = (2 * torch.stack([position.unsqueeze(1) * c for position in x]))
        out = input_const
    # full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
    return out

def piecewise_value(x, sig_m, sig_c, cavex_m, cavex_c, old=False):
    if old:
        sy = []
        cavexy = []
        for sm, sc, cvm, cvc in zip(sig_m, sig_c, cavex_m, cavex_c):
            smx = torch.matmul(x, sm)
            cavexx = torch.matmul(x, cvm)
            sy.append(smx + sc)
            cavexy.append(cavexx + cvc)
        sy = torch.stack(sy)
        cavexy = torch.stack(cavexy)
        max_vex_max3 = torch.max(cavexy, dim=0)[1]
        # min_cave_min3 = torch.min(-cavexy, dim=0)[1]
        vex_sy = torch.stack([torch.stack(
            [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3)])
        # cave_sy = torch.stack([torch.stack(
        #     [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
        # y = (vex_sy + cave_sy) / 2
        y = vex_sy
    else:
        vexy = []
        cavey = []
        for sm, sc, cvm, cvc in zip(sig_m, sig_c, cavex_m, cavex_c):
            vexx = torch.matmul(x, sm + cvm)
            cavex = torch.matmul(x, sm - cvm)
            vexy.append(vexx + cvc + sc)
            cavey.append(cavex - cvc + sc)
        vexy = torch.stack(vexy)
        cavey = torch.stack(cavey)
        y_max = torch.max(vexy, dim=0)[0]
        y_min = torch.min(cavey, dim=0)[0]
        y = (y_min + y_max) / 2
    return y

full_sig = []
full_cavex = []
old_full_cavex = []
full_sig_der = []
full_cavex_der = []
old_full_cavex_der = []
full_sig_c = []
full_cavex_c = []
old_full_cavex_c = []

legendre_scale_factor = 1#0e3

hidden_size = model.layer[0].bias.shape[0]


print("extracting Legendre transform")

for images, labels in tqdm(train_loader):
    images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
    for i, (w, b, out_w) in enumerate(zip(model.layer[0].weight.data,
                                   model.layer[0].bias.data,
                                   torch.transpose(model.layer[1].weight.data, 0, 1)
                                   )):

        w = w.type(default_type)
        b = b.type(default_type)
        out_w = out_w.type(default_type)

        neuron_sig = Sig(images, w, b, out_w)
        neuron_cavex = CaVexSig(images, w, out_w)
        old_neuron_cavex = CaVexSig(images, w, out_w, old=True)
        neuron_sig_der = Der(images, w, b, out_w, 'sig')
        neuron_cavex_der = Der(images, w, b, out_w, False)
        old_neuron_cavex_der = Der(images, w, b, out_w, False, old=True)

        for output, ow in enumerate(out_w):
            if ow < 0:
                neuron_cavex[:, output] *= -1
                neuron_cavex_der[:, :, output] *= -1
                old_neuron_cavex[:, output] *= -1
                old_neuron_cavex_der[:, :, output] *= -1

        if not i:
            full_sig.append(neuron_sig)
            full_cavex.append(neuron_cavex)
            old_full_cavex.append(old_neuron_cavex)
            full_sig_der.append(neuron_sig_der)
            full_cavex_der.append(neuron_cavex_der)
            old_full_cavex_der.append(old_neuron_cavex_der)
        else:
            full_sig[-1] += neuron_sig
            full_cavex[-1] += neuron_cavex
            old_full_cavex[-1] += old_neuron_cavex
            full_sig_der[-1] += neuron_sig_der
            full_cavex_der[-1] += neuron_cavex_der
            old_full_cavex_der[-1] += old_neuron_cavex_der

    full_sig_c.append(full_sig[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(full_sig_der[-1], images)]))
    full_cavex_c.append(full_cavex[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(full_cavex_der[-1], images)]))
    old_full_cavex_c.append(old_full_cavex[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(old_full_cavex_der[-1], images)]))
    print('', end='')

full_sig_c = torch.vstack(full_sig_c)
full_cavex_c = torch.vstack(full_cavex_c)
old_full_cavex_c = torch.vstack(old_full_cavex_c)
full_sig_der = torch.vstack(full_sig_der)
full_cavex_der = torch.vstack(full_cavex_der)
old_full_cavex_der = torch.vstack(old_full_cavex_der)

print('', end='')


torch.save(full_sig_c, 'data/sig_c {}.pt'.format(net_file))
torch.save(full_sig_der, 'data/sig_der {}.pt'.format(net_file))
torch.save(full_cavex_c, 'data/x2_cavex_c {}.pt'.format(net_file))
torch.save(full_cavex_der, 'data/x2_cavex_der {}.pt'.format(net_file))
torch.save(old_full_cavex_c, 'data/old_cavex_c {}.pt'.format(net_file))
torch.save(old_full_cavex_der, 'data/old_cavex_der {}.pt'.format(net_file))
# print("Loading data")
# full_sig_c = torch.load('data/sig_c {}.pt'.format(net_file))
# full_sig_der = torch.load('data/sig_der {}.pt'.format(net_file))
# full_cavex_c = torch.load('data/x2_cavex_c {}.pt'.format(net_file))
# full_cavex_der = torch.load('data/x2_cavex_der {}.pt'.format(net_file))
# old_full_cavex_c = torch.load('data/old_cavex_c {}.pt'.format(net_file))
# old_full_cavex_der = torch.load('data/old_cavex_der {}.pt'.format(net_file))

with torch.no_grad():
    correct_m = 0
    old_correct_xtx = 0
    old_loss_xtx = 0
    correct_xtx = 0
    loss_xtx = 0
    old_correct_x2 = 0
    old_loss_x2 = 0
    correct_x2 = 0
    loss_x2 = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_old_xtx = piecewise_value(images, full_sig_der, full_sig_c, full_cavex_der, full_cavex_c, old=True)
        _, pred = torch.max(out_old_xtx, 1)
        old_correct_xtx += (pred == labels).sum().item()
        old_loss_xtx += torch.sum(torch.abs(out_old_xtx - out_m))
        
        out_xtx = piecewise_value(images, full_sig_der, full_sig_c, full_cavex_der, full_cavex_c, old=False)
        _, pred = torch.max(out_xtx, 1)
        correct_xtx += (pred == labels).sum().item()
        loss_xtx += torch.sum(torch.abs(out_xtx - out_m))
        
        out_old_x2 = piecewise_value(images, full_sig_der, full_sig_c, old_full_cavex_der, old_full_cavex_c, old=True)
        _, pred = torch.max(out_old_x2, 1)
        old_correct_x2 += (pred == labels).sum().item()
        old_loss_x2 += torch.sum(torch.abs(out_old_x2 - out_m))
        
        out_x2 = piecewise_value(images, full_sig_der, full_sig_c, old_full_cavex_der, old_full_cavex_c, old=False)
        _, pred = torch.max(out_x2, 1)
        correct_x2 += (pred == labels).sum().item()
        loss_x2 += torch.sum(torch.abs(out_x2 - out_m))

        total += labels.size(0)
        print("Current total {}".format(total))
        print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Old xTx Legendre testing accuracy: {} %'.format(100 * old_correct_xtx / total))
        print('Old xTx Average Legendre loss: {}'.format(old_loss_xtx / total))
        print('New xTx Legendre testing accuracy: {} %'.format(100 * correct_xtx / total))
        print('New xTx Average Legendre loss: {}'.format(loss_xtx / total))
        print('Old x^2 Legendre testing accuracy: {} %'.format(100 * old_correct_x2 / total))
        print('Old x^2 Average Legendre loss: {}'.format(old_loss_x2 / total))
        print('New x^2 Legendre testing accuracy: {} %'.format(100 * correct_x2 / total))
        print('New x^2 Average Legendre loss: {}'.format(loss_x2 / total))
    # testing_accuracies = 100 * np.array(correct) / total


print("Done")
