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
# net_file = 'corner sigmoid nosoftorbias hidden_size[1] test_acc[78.90625].pt'
# net_file = 'corner sigmoid nosoftorbias hidden_size[3] test_acc[98.90625].pt'
# net_file = 'data/xor sigmoid nosoftorbias hidden_size[8] test_acc[99.53125].pt'
net_file = 'data/xor relu nosoftorbias hidden_size[8] test_acc[99.6875].pt'
model = torch.load(net_file)

def act(x, w, b, out_w):
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

def piecewise_value(x, sig_m, sig_c, cavex_m, cavex_c, soft=False):
    sy = []
    vexy = []
    cavey = []
    cavexy = []
    for sm, sc, cvm, cvc in zip(sig_m, sig_c, cavex_m, cavex_c):
        smx = torch.matmul(x, sm)
        vexx = torch.matmul(x, sm + cvm)
        cavex = torch.matmul(x, sm - cvm)
        cavexx = torch.matmul(x, cvm)
        sy.append((smx + sc) / legendre_scale_factor)
        vexy.append((vexx + cvc + sc) / legendre_scale_factor)
        cavey.append((cavex - cvc + sc) / legendre_scale_factor)
        cavexy.append(cavexx + cvc)
    sy = torch.stack(sy)
    vexy = torch.stack(vexy)
    cavey = torch.stack(cavey)
    cavexy = torch.stack(cavexy)
    max_vex_max = torch.max(vexy, dim=0)[1]
    min_cave_min = torch.min(cavey, dim=0)[1]
    max_vex_max2 = torch.max(sy + cavexy, dim=0)[1]
    min_cave_min2 = torch.min(sy - cavexy, dim=0)[1]
    max_vex_max3 = torch.max(cavexy, dim=0)[1]
    min_cave_min3 = torch.min(-cavexy, dim=0)[1]
    vex_sy = torch.stack([torch.stack(
        [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3)])
    cave_sy = torch.stack([torch.stack(
        [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
    y = (vex_sy + cave_sy) / 2
    return y
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

full_sig = []
full_cavex = []
full_sig_der = []
full_cavex_der = []
full_sig_c = []
full_cavex_c = []

legendre_scale_factor = 1#0e3

hidden_size = model.layer[0].bias.shape[0]


print("extracting Legendre transform")
resolution = 5*8
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
for images, labels in tqdm(train_loader):
# for b_i in batch_indexes:
#     images = spanning_data[batch_indexes]
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
        neuron_sig = Sig(images, w, b, out_w)
        neuron_cavex = CaVexSig(images, w, out_w)
        neuron_sig_der = Der(images, w, b, out_w, 'sig')
        neuron_cavex_der = Der(images, w, b, out_w, False)

        for output, ow in enumerate(out_w):
            if ow < 0:
                # neuron_sig[output] *= -1
                # neuron_sig_der[output] *= -1
                neuron_cavex[:, output] *= -1
                neuron_cavex_der[:, :, output] *= -1

        if not i:
            full_sig.append(neuron_sig)
            full_cavex.append(neuron_cavex)
            full_sig_der.append(neuron_sig_der)
            full_cavex_der.append(neuron_cavex_der)
        else:
            full_sig[-1] += neuron_sig
            full_cavex[-1] += neuron_cavex
            full_sig_der[-1] += neuron_sig_der
            full_cavex_der[-1] += neuron_cavex_der

    # real_net = model(images)
    full_sig_c.append(full_sig[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(full_sig_der[-1], images)]))
    full_cavex_c.append(full_cavex[-1] - torch.stack(
        [torch.matmul(im, f) for f, im in zip(full_cavex_der[-1], images)]))
    print('', end='')

full_sig_c = torch.vstack(full_sig_c)
full_cavex_c = torch.vstack(full_cavex_c)
full_sig_der = torch.vstack(full_sig_der)
full_cavex_der = torch.vstack(full_cavex_der)

print('', end='')

# full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
# full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
# full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
# full_cave_legendre_c = torch.vstack(full_cave_legendre_c)
# full_vex_legendre_m = torch.load('vex_m.pt')
# full_vex_legendre_c = torch.load('vex_c.pt')
# full_cave_legendre_m = torch.load('cave_m.pt')
# full_cave_legendre_c = torch.load('cave_c.pt')

with torch.no_grad():
    correct_m = 0
    correct_l = 0
    total = 0
    for images, labels in tqdm(test_loader):
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_l = piecewise_value(images, full_sig_der, full_sig_c, full_cavex_der, full_cavex_c)

        _, pred = torch.max(out_l, 1)
        correct_l += (pred == labels).sum().item()

        total += labels.size(0)
        print("Current total {}/{}".format(total+1, len(test_loader)))
        print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
    # testing_accuracies = 100 * np.array(correct) / total

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
legendre_output = []
with torch.no_grad():
    for b_i in tqdm(batch_indexes):
        batch = spanning_data[b_i]
        model_output.append(model(batch))
        out_l = piecewise_value(batch, full_sig_der, full_sig_c, full_cavex_der, full_cavex_c)
        legendre_output.append(out_l)

model_output = torch.vstack(model_output).cpu().reshape(resolution, resolution, 2)
legendre_output = torch.vstack(legendre_output).cpu().reshape(resolution, resolution, 2)
spanning_data = spanning_data.cpu().reshape(resolution, resolution, 2)

print("plotting")
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 0],
                  color='green', alpha=1, label='model out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], model_output[:, :, 1],
                  color='red', alpha=1, label='model out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  model_output[:, :, 0] + model_output[:, :, 1],
                  color='blue', alpha=1, label='model output')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  legendre_output[:, :, 0] + legendre_output[:, :, 1],
                  color='green', alpha=1, label='Legendre output')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 3, 4, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 0],
                  color='green', alpha=1, label='Legendre out0')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 3, 5, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1], legendre_output[:, :, 1],
                  color='red', alpha=1, label='Legendre out1')
ax.legend(loc='lower right')

ax = fig.add_subplot(2, 3, 6, projection='3d')
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  legendre_output[:, :, 0] + legendre_output[:, :, 1],
                  color='blue', alpha=1, label='Legendre output')
ax.legend(loc='lower right')

plt.suptitle("Conversion of small 2in2out network", fontsize=16)
fig.subplots_adjust(wspace=0.08, hspace=0.015, left=0.015, bottom=0, right=0.98, top=1)
plt.show()

print("Done")
