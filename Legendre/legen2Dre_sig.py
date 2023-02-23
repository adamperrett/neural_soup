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

sns.set_theme(style="whitegrid")


def sigmoid(position, w, b):
    return 1 / (1 + np.exp(-np.sum(w * position) - b))


def vex_sigmoid(position, w, b):
    c = np.square(w) * 0.5
    return sigmoid(position, w, b) + np.sum(c * np.square(position))


def vex_der(position, w, b):
    c = np.square(w) * 0.5
    sig = sigmoid(position, w, b)
    return w * sig * (1 - sig) + (2 * c * position)


def cave_sigmoid(position, w, b):
    c = np.square(w) * 0.5
    return sigmoid(position, w, b) - np.sum(c * np.square(position))


def cave_der(position, w, b):
    c = np.square(w) * 0.5
    sig = sigmoid(position, w, b)
    return w * sig * (1 - sig) - (2 * c * position)


def piecewise_value(position, legendre_m, legendre_c, vex=True, soft=False):
    y = []
    for m, c in zip(legendre_m, legendre_c):
        y.append((np.sum(m * position) + c) / legendre_scale_factor)
    if soft:
        y = np.array(y)
        temperature = .01
        if vex:
            return np.sum(y * np.exp(y / temperature) / np.sum(np.exp(y / temperature)))
        else:
            return np.sum(y * np.exp(-y / temperature) / np.sum(np.exp(-y / temperature)))
    else:
        if vex:
            return np.max(y)
        else:
            return np.min(y)


g = [
    np.array([np.array([2.9534, 2.6969]), -0.1383])
]

np.random.seed(272727)

dimensions = 2
bias = 100
weighting = 40
n_sig = 5
# g = [[np.array([np.random.random() * weighting - (weighting / 2) for j in range(dimensions)]),
#       np.random.random() * bias - (bias / 2)]
#      for i in range(n_sig)]

v_min = -1
v_max = 1
increments = 5

full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []
random_chance = 0.3
l_amount = 279
legendre_scale_factor = 10e3

s_values = [[] for i in range(len(g))]
s_total = []
vex_values = [[] for i in range(len(g))]
vex_total = []
cave_values = [[] for i in range(len(g))]
cave_total = []
print("processing values")
coords = [np.linspace(v_min, v_max, increments) for i in range(dimensions)]
coords = list(itertools.product(*coords))

selected_c = np.array(coords)[np.linspace(0, len(coords) - 1, l_amount, dtype=int)]

for position in tqdm(coords):
    for i, (w, b) in enumerate(g):
        s_values[i].append(sigmoid(position, w, b))
        vex_values[i].append(vex_sigmoid(position, w, b))
        cave_values[i].append(cave_sigmoid(position, w, b))
        if not i:
            s_total.append(s_values[i][-1])
            full_vex_legendre_m.append(vex_der(position, w, b))
            vex_total.append(vex_values[i][-1])
            full_cave_legendre_m.append(cave_der(position, w, b))
            cave_total.append(cave_values[i][-1])
        else:
            s_total[-1] += s_values[i][-1]
            full_vex_legendre_m[-1] += vex_der(position, w, b)
            vex_total[-1] += vex_values[i][-1]
            full_cave_legendre_m[-1] += cave_der(position, w, b)
            cave_total[-1] += cave_values[i][-1]

    full_vex_legendre_c.append(vex_total[-1] - np.sum(full_vex_legendre_m[-1] * position))
    full_cave_legendre_c.append(cave_total[-1] - np.sum(full_cave_legendre_m[-1] * position))
    # if np.random.random() > random_chance:
    # if x < legendre_min or x > legendre_max:
    #     if np.random.random() > random_chance * (x_max - x_min) / (legendre_max - legendre_min):
    # if np.random.random() > random_chance:
    if list(position) not in selected_c.tolist():
        del full_vex_legendre_m[-1]
        del full_vex_legendre_c[-1]
        del full_cave_legendre_m[-1]
        del full_cave_legendre_c[-1]

test_label = '{}L {}Dx{} s({}+{})x{}'.format(
    len(full_vex_legendre_c), dimensions, increments, weighting, bias, n_sig)

legendre_values_total = []
legendre_values_vex = []
legendre_values_cave = []
legendre_x = coords
print("extracting values from", len(full_vex_legendre_c), "Legendre planes")
for position in tqdm(legendre_x):
    legendre_values_vex.append(piecewise_value(position, full_vex_legendre_m, full_vex_legendre_c))
    legendre_values_cave.append(piecewise_value(position, full_cave_legendre_m, full_cave_legendre_c, vex=False))
    legendre_values_total.append(
        (legendre_values_vex[-1] + legendre_values_cave[-1]) * (legendre_scale_factor / 2))

# print("collecting planes")
# plane_collection = []
# for m, c in zip(full_vex_legendre_m, full_vex_legendre_c):
#     y = []
#     for position in coords:
#         y.append(np.sum(m*position) + c)
#     plane_collection.append(y)
# for m, c in zip(full_cave_legendre_m, full_cave_legendre_c):
#     y = []
#     for position in coords:
#         y.append(np.sum(m*position) + c)
#     plane_collection.append(y)

print("plotting")
fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = fig.add_subplot(1, 2, 1, projection='3d')
shape = [increments for i in range(dimensions)]
shape.append(dimensions)
cropped_cords = np.reshape(coords, shape)
while len(cropped_cords.shape) > 3:
    cropped_cords = cropped_cords[0]

# for plane in plane_collection:
#     plane = np.reshape(plane, [increments for i in range(dimensions)])
#     plotting_p = np.array(plane)
#     while len(plotting_p.shape) > 2:
#         plotting_p = plotting_p[0]
#     ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_p, color='black', alpha=0.05)

vex_total = np.reshape(vex_total, [increments for i in range(dimensions)])
plotting_v = np.array(vex_total)
cave_total = np.reshape(cave_total, [increments for i in range(dimensions)])
plotting_c = np.array(cave_total)
while len(plotting_v.shape) > 2:
    plotting_v = plotting_v[0]
    plotting_c = plotting_c[0]

# ax.plot_surface(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_v, cmap='viridis')#, 'r--', label='sigmoid_total')
# ax.plot_surface(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_c, cmap='viridis')#, 'r--', label='sigmoid_total')
# ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_v, color='green')#, 'r--', label='sigmoid_total')
# ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_c, color='green')#, 'r--', label='sigmoid_total')

# ax1.plot(x, vex_total, label='vex_total')
# ax1.plot(x, cave_total, label='cave_total')

# ax1.plot(legendre_x, legendre_values_vex, label='legendre_values_vex')
# ax1.plot(legendre_x, legendre_values_cave, label='legendre_values_cave')
# ax1.plot(legendre_x, legendre_values_total, 'k', label='legendre_values_total')
legendre_values_vex = np.reshape(legendre_values_vex, [increments for i in range(dimensions)])
legendre_values_cave = np.reshape(legendre_values_cave, [increments for i in range(dimensions)])
legendre_values_total = np.reshape(legendre_values_total, [increments for i in range(dimensions)])
plotting_lv = np.array(legendre_values_vex)
plotting_lc = np.array(legendre_values_cave)
plotting_l = np.array(legendre_values_total)
while len(plotting_l.shape) > 2:
    plotting_l = plotting_l[0]
# ax.plot_surface(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_l, cmap='viridis')#, 'r--', label='sigmoid_total')
# ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_lv,
#                   color='blue', alpha=0.5, label='Legendre_vex')
# ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_lc,
# color='blue', alpha=0.5, label='Legendre_cave')
ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_l,
                  color='green', alpha=0.5, label='Legendre_total')

s_total = np.reshape(s_total, [increments for i in range(dimensions)])
plotting_t = np.array(s_total)
while len(plotting_t.shape) > 2:
    plotting_t = plotting_t[0]
# ax.plot_surface(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_t, cmap='viridis')#, 'r--', label='sigmoid_total')
ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_t,
                  color='red', alpha=0.5, label='sigmoid_total')

ax.legend(loc='lower right')

plotting_d = plotting_l - plotting_t
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(cropped_cords[:, :, -2], cropped_cords[:, :, -1], plotting_d,
                  color='red', alpha=0.5, label='error')

ax.legend(loc='lower right')
# ax.set_ylim([-2, len(g) * (narrowness / spread) * 2])
# ax.set_zlim([-1, 1])
# ax1.set_ylim([-2, len(g) + 1])
# ax2.set_ylim([0, 2])
plt.suptitle(test_label, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

plt.show()

print("done")
