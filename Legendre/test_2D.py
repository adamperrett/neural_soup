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
from Legendre.train_2D import generate_corner_2class_data

sns.set_theme(style="whitegrid")

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

torch.manual_seed(272727)

# batch_size = 3
# train_loader = generate_corner_2class_data(batches=5, batch_size=batch_size)
# test_loader = generate_corner_2class_data(batches=3, batch_size=batch_size)

batch_size = 64
train_loader = generate_corner_2class_data(batches=40, batch_size=batch_size)
test_loader = generate_corner_2class_data(batches=10, batch_size=batch_size)

net_file = 'corner sigmoid hidden_size[8] test_acc[90.9375].pt'
model = torch.load(net_file)

def Sig(x, w, b, out_w):
        out = out_w.unsqueeze(1) / (1 + torch.sum(torch.exp(-(x * w)) + b, dim=1))
        return torch.transpose(out, 0, 1)

def VexSig(x, w, b, out_w):
        c = torch.square(w) * 0.5 * out_w
        out = Sig(x, w, b, out_w) + torch.sum(c * torch.square(x), dim=1).unsqueeze(1)
        return out

def VexDer(x, w, b, out_w):
        c = torch.square(w) * 0.5 * out_w
        sig = Sig(x, w, b, out_w)
        out = w * sig * (1 - sig) + (2 * c * x)
        full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
        return full_out

def CaveSig(x, w, b, out_w):
        c = torch.square(w) * 0.5 * out_w
        out = Sig(x, w, b, out_w) - torch.sum(c * torch.square(x), dim=1).unsqueeze(1)
        return out

def CaveDer(x, w, b, out_w):
        c = torch.square(w) * 0.5 * out_w
        sig = Sig(x, w, b, out_w)
        out = w * sig * (1 - sig) - (2 * c * x)
        full_out = torch.matmul(out.unsqueeze(2), out_w.unsqueeze(0))
        return full_out


def piecewise_value(x, legendre_m, legendre_c, vex=True, soft=False):
    y = []
    for m, c in zip(legendre_m, legendre_c):
        mx = torch.hstack([m_i * x for m_i in m]).reshape(x.shape[0], m.shape[0], m.shape[1])
        y.append((torch.sum(mx, dim=1) + c) / legendre_scale_factor)
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
net_values = [[] for i in range(hidden_size)]
net_total = []
vex_values = [[] for i in range(hidden_size)]
vex_total = []
cave_values = [[] for i in range(hidden_size)]
cave_total = []

print("extracting Legendre transform")
out_b = model.layer[1].bias.data
for images, labels in train_loader:
    for i, (w, b, out_w) in enumerate(zip(model.layer[0].weight.data,
                                   model.layer[0].bias.data,
                                   torch.transpose(model.layer[1].weight.data, 0, 1)
                                   )):

        net_values[i].append(Sig(images, w, b, out_w))
        vex_values[i].append(VexSig(images, w, b, out_w))
        cave_values[i].append(CaveSig(images, w, b, out_w))
        if not i:
            net_total.append(net_values[i][-1])
            full_vex_legendre_m.append(VexDer(images, w, b, out_w))
            vex_total.append(vex_values[i][-1])
            full_cave_legendre_m.append(CaveDer(images, w, b, out_w))
            cave_total.append(cave_values[i][-1])
        else:
            net_total[-1] += net_values[i][-1]
            full_vex_legendre_m[-1] += VexDer(images, w, b, out_w)
            vex_total[-1] += vex_values[i][-1]
            full_cave_legendre_m[-1] += CaveDer(images, w, b, out_w)
            cave_total[-1] += cave_values[i][-1]

    full_vex_legendre_c.append(vex_total[-1] - torch.sum(torch.stack(
        [f*i for f, i in zip(full_vex_legendre_m[-1], images)]), dim=2))
    full_cave_legendre_c.append(cave_total[-1] - torch.sum(torch.stack(
        [f*i for f, i in zip(full_cave_legendre_m[-1], images)]), dim=2))

full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
full_cave_legendre_c = torch.vstack(full_cave_legendre_c)

print("extracting positional values")
resolution = 8 * 5
spanning_data = torch.stack([
    torch.stack([torch.tensor([x, y]) for x in np.linspace(-1, 1, resolution)])
    for y in np.linspace(-1, 1, resolution)]
).reshape(resolution**2, 2).type(torch.float32)

a_i = [i for i in range(resolution**2)]
batch_indexes = [a_i[j*batch_size:(j+1)*batch_size] for j in range(int(np.ceil(resolution**2/batch_size)))]

model_output = []
legendre_values_vex = []
legendre_values_cave = []
legendre_output = []
with torch.no_grad():
    for b_i in batch_indexes:
        batch = spanning_data[b_i]
        model_output.append(model(batch))
        legendre_values_vex.append(piecewise_value(batch, full_vex_legendre_m, full_vex_legendre_c))
        legendre_values_cave.append(piecewise_value(batch, full_cave_legendre_m, full_cave_legendre_c, vex=False))
        legendre_output.append((legendre_values_vex[-1] + legendre_values_cave[-1]) * (legendre_scale_factor / 2))

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

ax = fig.add_subplot(2, 4, 1, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 0],
                  color='green', alpha=1, label='model out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 2, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 0],
                  color='red', alpha=1, label='Legendre out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 3, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_vex[:, :, 0],
                  color='blue', alpha=1, label='Legendre vex out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 4, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_cave[:, :, 0],
                  color='purple', alpha=1, label='Legendre cave out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 5, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 1],
                  color='green', alpha=1, label='model out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 6, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 1],
                  color='red', alpha=1, label='Legendre out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 7, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_vex[:, :, 1],
                  color='blue', alpha=1, label='Legendre vex out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 4, 8, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_values_cave[:, :, 1],
                  color='purple', alpha=1, label='Legendre cave out1')
ax.legend(loc='lower right')

plt.suptitle("Conversion of small 2in2out network", fontsize=16)
fig.subplots_adjust(wspace=0.08, hspace=0.015, left=0.015, bottom=0, right=0.98, top=1)
plt.show()

print("Done")
