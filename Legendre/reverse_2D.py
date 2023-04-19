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
# net_file = 'xor relu nosoftorbias hidden_size[8] test_acc[99.6875]'
model = torch.load('data/'+net_file+'.pt')

num_outputs = 2

def calculate_c(model, all_x):
    net_hessian = torch.stack([torch.stack([
        torch.autograd.functional.hessian(model.separate_outputs(out), (all_x[i:i + 1])).squeeze()
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    eigen_values = torch.stack([torch.stack([
        torch.linalg.eig(net_hessian[i][out])[0].real
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    margin_of_error = 0.90001

    vex_c = torch.stack([torch.stack([
        torch.max(
            -torch.min(torch.hstack([
                eigen_values[i][out], torch.tensor(0)]))) + margin_of_error
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    cave_c = torch.stack([torch.stack([
        torch.min(
            -torch.max(torch.hstack([
                eigen_values[i][out], torch.tensor(0)]))) + margin_of_error
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    abs_c = torch.stack([torch.stack([
        torch.max(
            torch.abs(torch.hstack([
                eigen_values[i][out], torch.tensor(0)]))) + margin_of_error
        for out in range(num_outputs)])
        for i in range(all_x.shape[0])])

    output_vex = torch.max(vex_c, dim=0)[0]
    output_cave = torch.max(cave_c, dim=0)[0]
    output_abs = torch.max(abs_c, dim=0)[0]

    # return output_vex, output_cave, output_abs
    return output_abs

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
    max_vex_max3 = torch.max(cavexy, dim=2)
    min_cave_min3 = torch.min(-cavexy, dim=2)[1]
    vex_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max3[1])])
    cave_sy = torch.stack([torch.stack(
        [sy[j][i][output_i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min3)])
    if max_mindex:
        y = torch.transpose((vex_sy + cave_sy) / 2, 0, 1)
    else:
        y = torch.transpose((y_min + y_max) / 2, 0, 1)
    return y

def test_sample(x, net_m, net_c, cavex_m, cavex_c, population):
    vexy = []
    cavey = []
    for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
        vexx = torch.matmul(x, torch.transpose(sm + cvm, 0, 1))
        cavex = torch.matmul(x, torch.transpose(sm - cvm, 0, 1))
        vexy.append((vexx + cvc + sc))
        cavey.append((cavex - cvc + sc))
    vexy = torch.stack(vexy)
    cavey = torch.stack(cavey)
    all_y = []
    for agent in population:
        y_vex = torch.max(vexy[:, :, agent.cpu().numpy()], dim=2)[0]
        y_cave = torch.min(cavey[:, :, agent.cpu().numpy()], dim=2)[0]
        all_y.append(torch.transpose((y_vex + y_cave) / 2, 0, 1))
    return all_y

def reverse_legendre():
    agent_sample = np.sort(random_sample[best_agent].cpu().numpy())
    reduced_nm = net_m[:, agent_sample, :]
    reduced_nc = net_c[:, agent_sample]
    reduced_vm = cavex_m[:, agent_sample, :]
    reduced_vc = cavex_c[:, agent_sample]
    sample_out = net_out[:, agent_sample]
    sample_cavex = cavex_out[:, agent_sample]

    vex_m = reduced_nm + reduced_vm
    vex_c = reduced_nc + reduced_vc
    cave_m = reduced_nm - reduced_vm
    cave_c = reduced_nc - reduced_vc

    x_per_output = []
    out_comparison = []
    for nm, nc, vm, vc, so, sc in zip(reduced_nm, reduced_nc, reduced_vm, reduced_vc, sample_out, sample_cavex):
        '''
        y*2
        a = y - cvc
        x = a / cvm 
        '''
        vex_m = torch.transpose(nm + vm, 0, 1)
        vex_c = nc + vc
        cave_m = torch.transpose(nm - vm, 0, 1)
        cave_c = nc - vc
        y = so #- sc
        vex_x0 = (y - vex_c) / vex_m
        cave_x0 = (y - cave_c) / cave_m
        vex_xy = (2*y - vex_c) / vex_m
        cave_xy = (2*y - cave_c) / cave_m
        net_x0 = (y - nc) / torch.transpose(nm, 0, 1)
        vex_x1 = (y - vc) / torch.transpose(vm, 0, 1)
        cave_x1 = (y + vc) / torch.transpose(-vm, 0, 1)
        vex_x2 = (y - cave_c) / vex_m
        cave_x2 = (y - vex_c) / cave_m
        spanning_data[agent_sample] = spanning_data[agent_sample]
        x_per_output.append([vex_x0, cave_x0])
        out_comparison.append([
            torch.transpose(vex_x0, 0, 1),
            torch.transpose(cave_x0, 0, 1),
            torch.transpose((vex_x0 + cave_x0) / 2, 0, 1),
            spanning_data[agent_sample],
            piecewise_value(torch.transpose((vex_x0 + cave_x0) / 2, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False),
            torch.transpose(sample_out, 0, 1),
            piecewise_value(torch.transpose((vex_x0 + cave_x0) / 2, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False) - torch.transpose(sample_out, 0, 1),
            piecewise_value(spanning_data[agent_sample],
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=True),
            piecewise_value(spanning_data[agent_sample],
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=True) - torch.transpose(sample_out, 0, 1),
            piecewise_value(spanning_data[agent_sample],
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False),
            piecewise_value(spanning_data[agent_sample],
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False) - torch.transpose(sample_out, 0, 1),
            piecewise_value(torch.transpose(vex_x0, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False),
            piecewise_value(torch.transpose(vex_x0, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False) - torch.transpose(sample_out, 0, 1),
            piecewise_value(torch.transpose(cave_x0, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False),
            piecewise_value(torch.transpose(cave_x0, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False) - torch.transpose(sample_out, 0, 1),
            piecewise_value(torch.transpose(vex_xy + cave_xy, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False),
            piecewise_value(torch.transpose(vex_xy + cave_xy, 0, 1),
                            reduced_nm, reduced_nc,
                            reduced_vm, reduced_vc,
                            max_mindex=False) - torch.transpose(sample_out, 0, 1)])
        out_comparison[-1].append(((out_comparison[-1][3] + out_comparison[-1][5])/2) - out_comparison[-1][1])
        out_comparison[0][3][0] += 0.0001
        x = -(vex_c + cave_c) / (vex_m + cave_m)
        # torch.matmul(-(vex_c + cave_c), torch.transpose(torch.pinverse((vex_m + cave_m)), 1, 2))
        # plane = torch.matmul(x, torch.transpose((vex_m + cave_m), 0, 1))
        # piecewise_value(torch.transpose(vex_x0 + cave_x0, 0, 1), reduced_nm, reduced_nc, reduced_vm, reduced_vc,
                        # max_mindex=False) - torch.transpose(sample_out, 0, 1)
    return x_per_output



net_out = []
net_m = []
net_c = []
cavex_out = []
cavex_m = []
cavex_c = []

hidden_size = model.layer[0].bias.shape[0]

print("extracting Legendre transform")
resolution = 3#5*8
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

# with torch.no_grad():
#     print("calculating Legendre c")
#     c_list = []
#     for images, labels in tqdm(train_loader):
#     # for b_i in tqdm(batch_indexes):
#     #     images = spanning_data[b_i]
#         cavex_const = calculate_c(model, images)
#         c_list.append(cavex_const)
#     cavex_const = torch.max(torch.stack(c_list), dim=0)[0]
#     # cavex_const = torch.ones(num_outputs) * 50
cavex_const = torch.tensor([31.6234, 32.9373])

print("calculating Legendre planes")
# for images, labels in tqdm(train_loader):
for b_i in tqdm(batch_indexes):
    images = spanning_data[b_i]
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

net_out = torch.hstack(net_out)
net_m = torch.hstack(net_m)
net_c = torch.hstack(net_c)
cavex_out = torch.hstack(cavex_out)
cavex_m = torch.hstack(cavex_m)
cavex_c = torch.hstack(cavex_c)

print('', end='')
out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c)

# full_vex_legendre_m = torch.vstack(full_vex_legendre_m)
# full_vex_legendre_c = torch.vstack(full_vex_legendre_c)
# full_cave_legendre_m = torch.vstack(full_cave_legendre_m)
# full_cave_legendre_c = torch.vstack(full_cave_legendre_c)
# full_vex_legendre_m = torch.load('vex_m.pt')
# full_vex_legendre_c = torch.load('vex_c.pt')
# full_cave_legendre_m = torch.load('cave_m.pt')
# full_cave_legendre_c = torch.load('cave_c.pt')

sample_size = 100
genome_length = len(spanning_data)
random_sample = torch.stack([torch.tensor(
    np.random.choice(range(net_c.shape[1]), genome_length, replace=False)) for i in range(sample_size)])
correct_m = 0
correct_l = [0 for i in range(sample_size)]
total = 0
for batch, (images, labels) in enumerate(tqdm(test_loader)):
    out_m = model(images.type(torch.float32))
    _, pred = torch.max(out_m, 1)
    correct_m += (pred == labels).sum().item()
    out_l = test_sample(images, net_m, net_c, cavex_m, cavex_c, random_sample)
    for i, out in enumerate(out_l):
        _, pred = torch.max(out, 1)
        correct_l[i] += (pred == labels).sum().item()
    total += labels.size(0)
print('Model testing accuracy: {} %'.format(100 * correct_m / total))
print("Max = {}   min = {}   average = {}".format(
    np.max(100 * np.array(correct_l) / total),
    np.min(100 * np.array(correct_l) / total),
    np.mean(100 * np.array(correct_l) / total),
))
best_score = 0
best_agent = 5e100
for agent, score in enumerate(correct_l):
    if score > best_score:
        best_score = score
        best_agent = agent

reverse_legendre()


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

        all_m.append(out_m)
        all_l.append(out_l)
        all_mindex.append(mindex_out_l)
# print("Current total {}".format(total))
print('Model testing accuracy: {} %'.format(100 * correct_m / total))
print('Legendre testing accuracy: {} %'.format(100 * correct_l / total))
print('Mindex Legendre testing accuracy: {} %'.format(100 * correct_mindex / total))
all_m = torch.vstack(all_m)
all_l = torch.vstack(all_l)
all_mindex = torch.vstack(all_mindex)
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

        all_m.append(out_m)
        all_l.append(out_l)
        all_mindex.append(mindex_out_l)
# print("Current total {}".format(total))
print('Model training accuracy: {} %'.format(100 * correct_m / total))
print('Legendre training accuracy: {} %'.format(100 * correct_l / total))
print('Mindex Legendre training accuracy: {} %'.format(100 * correct_mindex / total))
all_m = torch.vstack(all_m)
all_l = torch.vstack(all_l)
all_mindex = torch.vstack(all_mindex)
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
                  color='red', alpha=1, label='Mindex Legendre output')
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
