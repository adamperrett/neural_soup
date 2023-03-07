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
# net_file = 'corner sigmoid nosoftorbias hidden_size[1] test_acc[78.90625]'
# net_file = 'corner sigmoid nosoftorbias hidden_size[3] test_acc[98.90625]'
# net_file = 'xor sigmoid nosoftorbias hidden_size[8] test_acc[99.53125]'
net_file = 'xor sigmoid separate_out hidden_size[8] test_acc[99.53125]'
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

def calculate_c(model, all_x):
    net_hessian = torch.stack([torch.stack([
        torch.autograd.functional.hessian(model.output_0, (all_x[i:i + 1])).squeeze(),
        torch.autograd.functional.hessian(model.output_1, (all_x[i:i + 1])).squeeze()])
        for i in range(all_x.shape[0])])

    eigen_values = torch.stack([torch.stack([
        torch.linalg.eig(net_hessian[i][0])[0].real,
        torch.linalg.eig(net_hessian[i][1])[0].real])
        for i in range(all_x.shape[0])])

    margin_of_error = 0.0001
    cavex_c = torch.stack([torch.stack([
        torch.max(
            -torch.min(torch.hstack([
                eigen_values[i][0], torch.tensor(0)]))) + margin_of_error,
        torch.max(
            -torch.min(torch.hstack([
                eigen_values[i][1], torch.tensor(0)]))) + margin_of_error])
        for i in range(all_x.shape[0])])

    output_c = torch.max(cavex_c, dim=0)

    return output_c[0]

def piecewise_value(x, net_m, net_c, cavex_m, cavex_c, max_mindex=False):
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
    max_vex_max = torch.max(vexy, dim=2)[1]
    min_cave_min = torch.min(cavey, dim=2)[1]
    max_vex_max2 = torch.max(sy + cavexy, dim=2)[1]
    min_cave_min2 = torch.min(sy - cavexy, dim=2)[1]
    max_vex_max3 = torch.max(cavexy, dim=2)[1]
    min_cave_min3 = torch.min(-cavexy, dim=2)[1]
    vex_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3)])
    cave_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
    if max_mindex:
        y = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    else:
        y = torch.transpose((y_min + y_max) / 2, 0, 1)
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

net_out = []
net_m = []
net_c = []
cavex_out = []
cavex_m = []
cavex_c = []

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

with torch.no_grad():
    print("calculating Legendre c")
    c_list = []
    for images, labels in tqdm(train_loader):
    # for b_i in tqdm(batch_indexes):
    #     images = spanning_data[b_i]
        cavex_const = calculate_c(model, images)
        c_list.append(cavex_const)
    cavex_const = torch.max(torch.stack(c_list), dim=0)[0]

for images, labels in tqdm(train_loader):
# for b_i in batch_indexes:
#     images = spanning_data[b_i]

    images.requires_grad = True
    net_out_0 = model.output_0(images)
    net_out_0.backward(torch.ones(images.shape[0]))
    net_grad_0 = images.grad.clone().detach()
    net_out_0.detach()
    images.grad = None
    net_out_1 = model.output_1(images)
    net_out_1.backward(torch.ones(images.shape[0]))
    net_grad_1 = images.grad.clone().detach()
    net_out_1.detach()
    images.requires_grad = False

    with torch.no_grad():
        x2_out = torch.sum(images * images, dim=1)
        cx2 = 0.5 * torch.stack([torch.sum(images * images, dim=1) * c for c in cavex_const])
        cx2_grad = torch.stack([images * c for c in cavex_const])

        net_out.append(torch.stack([net_out_0, net_out_1]))
        net_m.append(torch.stack([net_grad_0, net_grad_1]))
        net_c.append(torch.stack([
            net_out_0 - torch.sum(net_grad_0 * images, dim=1),
            net_out_1 - torch.sum(net_grad_1 * images, dim=1)
        ]))
        cavex_out.append(cx2)
        cavex_m.append(cx2_grad)
        cavex_c.append(cx2 - torch.sum(cx2_grad * images, dim=2))

net_out = torch.hstack(net_out)
net_m = torch.hstack(net_m)
net_c = torch.hstack(net_c)
cavex_out = torch.hstack(cavex_out)
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

with torch.no_grad():
    correct_m = 0
    correct_l = 0
    correct_mindex = 0
    total = 0
    all_m = []
    all_l = []
    all_mindex = []
    for images, labels in tqdm(test_loader):
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)
        _, pred = torch.max(out_l, 1)
        correct_l += (pred == labels).sum().item()

        mindex_out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c, max_mindex=True)
        _, pred = torch.max(mindex_out_l, 1)
        correct_mindex += (pred == labels).sum().item()

        total += labels.size(0)
        print("Current total {}".format(total))
        print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
        print('Mindex Legendre testing accuracy: {} %'.format(100 * correct_mindex / total))

        all_m.append(out_m)
        all_l.append(out_l)
        all_mindex.append(mindex_out_l)
all_m = torch.stack(all_m)
all_l = torch.stack(all_l)
all_mindex = torch.stack(all_mindex)
print("Legendre difference", torch.sum(torch.abs(all_m - all_l)))
print("Minmax Legendre difference", torch.sum(torch.abs(all_m - all_mindex)))

with torch.no_grad():
    correct_m = 0
    correct_l = 0
    correct_mindex = 0
    total = 0
    all_m = []
    all_l = []
    all_mindex = []
    for images, labels in tqdm(train_loader):
        out_m = model(images.type(torch.float32))
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()

        out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)
        _, pred = torch.max(out_l, 1)
        correct_l += (pred == labels).sum().item()

        mindex_out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c, max_mindex=True)
        _, pred = torch.max(mindex_out_l, 1)
        correct_mindex += (pred == labels).sum().item()

        total += labels.size(0)
        print("Current total {}".format(total))
        print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
        print('Mindex Legendre testing accuracy: {} %'.format(100 * correct_mindex / total))

        all_m.append(out_m)
        all_l.append(out_l)
        all_mindex.append(mindex_out_l)
all_m = torch.stack(all_m)
all_l = torch.stack(all_l)
all_mindex = torch.stack(all_mindex)
print("Legendre difference", torch.sum(torch.abs(all_m - all_l)))
print("Minmax Legendre difference", torch.sum(torch.abs(all_m - all_mindex)))

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
mindex_legendre_output = []
with torch.no_grad():
    for b_i in tqdm(batch_indexes):
        batch = spanning_data[b_i]
        model_output.append(model(batch))
        out_l = piecewise_value(batch, net_m, net_c, cavex_m, cavex_c)
        legendre_output.append(out_l)
        out_mindex = piecewise_value(batch, net_m, net_c, cavex_m, cavex_c, max_mindex=True)
        mindex_legendre_output.append(out_mindex)

model_output = torch.vstack(model_output).cpu().reshape(resolution, resolution, 2)
legendre_output = torch.vstack(legendre_output).cpu().reshape(resolution, resolution, 2)
mindex_legendre_output = torch.vstack(mindex_legendre_output).cpu().reshape(resolution, resolution, 2)
spanning_data = spanning_data.cpu().reshape(resolution, resolution, 2)

print("Legendre difference", torch.sum(torch.abs(model_output - legendre_output)))
print("Minmax Legendre difference", torch.sum(torch.abs(model_output - mindex_legendre_output)))

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
ax.plot_wireframe(spanning_data[:, :, 0], spanning_data[:, :, 1],
                  mindex_legendre_output[:, :, 0] + mindex_legendre_output[:, :, 1],
                  color='green', alpha=1, label='Mindex Legendre output')
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
