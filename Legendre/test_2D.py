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
# batch_size = 3
# train_loader = generate_corner_2class_data(batches=5, batch_size=batch_size)
# test_loader = generate_corner_2class_data(batches=3, batch_size=batch_size)
batch_size = 64
# train_loader = generate_corner_2class_data(batches=40, batch_size=batch_size)
# test_loader = generate_corner_2class_data(batches=10, batch_size=batch_size)
train_loader = generate_xor_data(batches=40, batch_size=batch_size)
test_loader = generate_xor_data(batches=10, batch_size=batch_size)

print("loading net")
net_file = 'data/corner sigmoid nosoftorbias hidden_size[1] test_acc[78.90625].pt'
# net_file = 'corner sigmoid nosoftorbias hidden_size[3] test_acc[98.90625].pt'
# net_file = 'xor sigmoid nosoftorbias hidden_size[8] test_acc[99.53125].pt'
model = torch.load(net_file)

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

def CaVexSig(x, w, b, out_w, vex):
    # c = torch.square(w) * 0.5 * out_w
    c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
    x2c = torch.matmul(torch.square(x), c)
    out_w_sign = (out_w > 0) + ((out_w < 0) * -1)
    x2c *= out_w_sign
    if vex:
        out = Sig(x, w, b, out_w) + x2c
    else:
        out = Sig(x, w, b, out_w) - x2c
    return out


def CaVexDer(x, w, b, out_w, vex):
    # c = torch.square(w) * 0.5 * out_w
    c = torch.matmul(torch.square(w).unsqueeze(1) * 0.05, out_w.unsqueeze(0))
    sig = 1 / (1 + torch.exp(-torch.sum(x * w, dim=1) - b))
    w_scale = torch.matmul(w.unsqueeze(1), out_w.unsqueeze(0))
    input_const = (2 * torch.stack([position.unsqueeze(1) * c for position in x]))
    sig_derivative = (sig * (1 - sig))
    out_w_sign = (out_w > 0) + ((out_w < 0) * -1)
    input_const *= (torch.ones([2, 2]) * out_w_sign)
    if vex:
        out = torch.stack([w_scale * der for der in sig_derivative]) + input_const
    else:
        out = torch.stack([w_scale * der for der in sig_derivative]) - input_const
    # full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
    return out

def piecewise_value(x, legendre_m, legendre_c, vex=True, soft=False):
    y = []
    for m, c in zip(legendre_m, legendre_c):
        # mx = torch.hstack([m_i * x for m_i in m]).reshape(x.shape[0], m.shape[0], m.shape[1])
        mx = torch.matmul(x, m)
        # y.append((torch.sum(mx, dim=1) + c) / legendre_scale_factor)
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


full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []
legendre_scale_factor = 1#0e3

hidden_size = model.layer[0].bias.shape[0]
# net_values = [[] for i in range(hidden_size)]
# net_total = []
# vex_values = [[] for i in range(hidden_size)]
vex_total = []
# cave_values = [[] for i in range(hidden_size)]
cave_total = []

print("extracting Legendre transform")
resolution = 3
# spanning_data = torch.stack([
#     torch.stack([torch.tensor([x, y]) for x in np.linspace(-1, 1, resolution)])
#     for y in np.linspace(-1, 1, resolution)]
# ).reshape(resolution**2, 2).type(torch.float32)
spanning_data = torch.stack([
    torch.stack([torch.tensor([x, y]) for x in np.linspace(-1, 1, resolution)])
    for y in np.linspace(-1, 1, resolution)]
).reshape(resolution**2, 2).type(torch.float32)
a_i = [i for i in range(resolution**2)]
batch_indexes = [a_i[j*batch_size:(j+1)*batch_size] for j in range(int(np.ceil(resolution**2/batch_size)))]
# out_b = model.layer[1].bias.data
# for images, labels in tqdm(train_loader):
for b_i in batch_indexes:
    images = spanning_data[batch_indexes]
    for i, (w, b, out_w) in enumerate(zip(model.layer[0].weight.data,
                                   model.layer[0].bias.data,
                                   torch.transpose(model.layer[1].weight.data, 0, 1)
                                   )):

        w = w.type(default_type)
        b = b.type(default_type)
        out_w = out_w.type(default_type)

        # net_values[i].append(Sig(images, w, b, out_w))
        # vex_values[i].append(CaVexSig(images, w, b, out_w, True))
        # cave_values[i].append(CaVexSig(images, w, b, out_w, False))
        neuron_vex_total = CaVexSig(images, w, b, out_w, True)
        neuron_cave_total = CaVexSig(images, w, b, out_w, False)
        neuron_vex_legendre_m = CaVexDer(images, w, b, out_w, True)
        neuron_cave_legendre_m = CaVexDer(images, w, b, out_w, False)

        # for output, ow in enumerate(out_w):
        #     if ow < 0:
        #         temp = neuron_vex_legendre_m[:, :, output].clone().detach()
        #         neuron_vex_legendre_m[:, :, output] = neuron_cave_legendre_m[:, :, output]
        #         neuron_cave_legendre_m[:, :, output] = temp
        #         temp = neuron_cave_total[:, output].clone().detach()
        #         neuron_cave_total[:, output] = neuron_vex_total[:, output]
        #         neuron_vex_total[:, output] = temp

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
    print('', end='')

full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
full_cave_legendre_c = torch.vstack(full_cave_legendre_c)

print('', end='')

# full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
# full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
# full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
# full_cave_legendre_c = torch.vstack(full_cave_legendre_c)
# full_vex_legendre_m = torch.load('vex_m.pt')
# full_vex_legendre_c = torch.load('vex_c.pt')
# full_cave_legendre_m = torch.load('cave_m.pt')
# full_cave_legendre_c = torch.load('cave_c.pt')

# all_vex_output = []
# all_cave_output = []
# with torch.no_grad():
#     correct_m = 0
#     correct_l = 0
#     total = 0
#     for images, labels in tqdm(train_loader):
#         out_m = model(images.type(torch.float32))
#         _, pred = torch.max(out_m, 1)
#         correct_m += (pred == labels).sum().item()
#
#         vex = piecewise_value(images, full_vex_legendre_m, full_vex_legendre_c)
#         cave = piecewise_value(images, full_vex_legendre_m, full_vex_legendre_c, vex=False)
#         out_l = (vex + cave) * legendre_scale_factor / 2
#         _, pred = torch.max(out_l, 1)
#         correct_l += (pred == labels).sum().item()
#
#         all_vex_output.append(vex)
#         all_cave_output.append(cave)
#
#         total += labels.size(0)
#         print("Current total {}/{}".format(total+1, len(train_loader)))
#         print('Model testing accuracy: {} %'.format(100 * correct_m / total))
#         print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
#     # testing_accuracies = 100 * np.array(correct) / total

print("extracting positional values")
# resolution = 3#5*8
# # spanning_data = torch.stack([
# #     torch.stack([torch.tensor([x, y]) for x in np.linspace(-1, 1, resolution)])
# #     for y in np.linspace(-1, 1, resolution)]
# # ).reshape(resolution**2, 2).type(torch.float32)
# spanning_data = torch.stack([
#     torch.stack([torch.tensor([x, y]) for x in np.linspace(-1, 1, resolution)])
#     for y in np.linspace(-1, 1, resolution)]
# ).reshape(resolution**2, 2).type(torch.float32)
#
# a_i = [i for i in range(resolution**2)]
# batch_indexes = [a_i[j*batch_size:(j+1)*batch_size] for j in range(int(np.ceil(resolution**2/batch_size)))]

model_output = []
legendre_values_vex = []
legendre_values_cave = []
legendre_output = []
with torch.no_grad():
    for b_i in tqdm(batch_indexes):
        batch = spanning_data[b_i]
        model_output.append(model(batch))
        legendre_values_vex.append(piecewise_value(batch, full_vex_legendre_m, full_vex_legendre_c))
        legendre_values_cave.append(piecewise_value(batch, full_cave_legendre_m, full_cave_legendre_c, vex=False))
        # legendre_output.append(
        #     torch.nn.functional.log_softmax(
        #         (legendre_values_vex[-1] + legendre_values_cave[-1]) * (legendre_scale_factor / 2), dim=1))
        legendre_output.append(
            (legendre_values_vex[-1] + legendre_values_cave[-1]) * (legendre_scale_factor / 2))

model_output = torch.vstack(model_output).cpu().reshape(resolution, resolution, 2)
legendre_values_vex = torch.vstack(legendre_values_vex).cpu().reshape(resolution, resolution, 2)
legendre_values_cave = torch.vstack(legendre_values_cave).cpu().reshape(resolution, resolution, 2)
legendre_output = torch.vstack(legendre_output).cpu().reshape(resolution, resolution, 2)
spanning_data = spanning_data.cpu().reshape(resolution, resolution, 2)

print("plotting")
fig = plt.figure()
# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 0],
#                   color='green', alpha=1, label='model out0')
# ax.legend(loc='lower right')
#
# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 1],
#                   color='green', alpha=1, label='model out1')
# ax.legend(loc='lower right')
#
# ax = fig.add_subplot(2, 2, 3, projection='3d')
# ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 0],
#                   color='red', alpha=1, label='Legendre out0')
# ax.legend(loc='lower right')
#
# ax = fig.add_subplot(2, 2, 4, projection='3d')
# ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 1],
#                   color='red', alpha=1, label='Legendre out1')
# ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 1, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 0],
                  color='green', alpha=1, label='model out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 2, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 0],
                  color='red', alpha=1, label='Legendre out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 3, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_vex[:, :, 0],
                  color='blue', alpha=1, label='Legendre vex out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 4, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_cave[:, :, 0],
                  color='purple', alpha=1, label='Legendre cave out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 5, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 1],
                  color='green', alpha=1, label='model out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 6, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 1],
                  color='red', alpha=1, label='Legendre out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 7, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_vex[:, :, 1],
                  color='blue', alpha=1, label='Legendre vex out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 8, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_cave[:, :, 1],
                  color='purple', alpha=1, label='Legendre cave out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 9, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  model_output[:, :, 0] + model_output[:, :, 1],
                  color='green', alpha=1, label='model out')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  legendre_output[:, :, 0] + legendre_output[:, :, 1],
                  color='red', alpha=1, label='Legendre out')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 10, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  legendre_output[:, :, 0] + legendre_output[:, :, 1],
                  color='red', alpha=1, label='Legendre out')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 11, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  legendre_values_vex[:, :, 0] + legendre_values_vex[:, :, 1],
                  color='blue', alpha=1, label='Legendre vex out')
ax.legend(loc='lower right')

ax = fig.add_subplot(3, 4, 12, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  legendre_values_cave[:, :, 0] + legendre_values_cave[:, :, 1],
                  color='purple', alpha=1, label='Legendre cave out')
ax.legend(loc='lower right')

plt.suptitle("Conversion of small 2in2out network", fontsize=16)
fig.subplots_adjust(wspace=0.08, hspace=0.015, left=0.015, bottom=0, right=0.98, top=1)
plt.show()

print("Done")
