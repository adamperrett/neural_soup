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
from tqdm import tqdm

sns.set_theme(style="whitegrid")


def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(w * x) + b))


def vex_sigmoid(x, w, b):
    c = np.square(w) * 0.05
    return 1 / (1 + np.exp(-(w * x) + b)) + c * (x ** 2)


def vex_der(x, w, b):
    c = np.square(w) * 0.05
    sig = sigmoid(x, w, b)
    return (w * sig * (1 - sig)) + (2 * c * x)


def cave_sigmoid(x, w, b):
    c = np.square(w) * 0.05
    return 1 / (1 + np.exp(-(w * x) + b)) - c * (x ** 2)


def cave_der(x, w, b):
    c = np.square(w) * 0.05
    sig = sigmoid(x, w, b)
    return (w * sig * (1 - sig)) - (2 * c * x)


def piecewise_value(x, legendre_m, legendre_c, vex=True, soft=True):
    y = []
    for m, c in zip(legendre_m, legendre_c):
        y.append((m * x + c) / legendre_scale_factor)
    if soft:
        y = np.array(y)
        temperature = 0.1
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
    # [0, 1],
    [10, 0]
]

bias = 100
weighting = 200
n_sig = 1
# g = [[np.random.random() * weighting - (weighting / 2), np.random.random() * bias - (bias / 2)]
#      for i in range(n_sig)]

# g = [[1, 0]]

x_min = -2
x_max = 2
increments = 100

full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []
random_chance = 0.03
l_amount = 5
selected_x = np.linspace(x_min, x_max, increments)[np.linspace(0, increments - 1, l_amount, dtype=int)]
legendre_min = x_min
legendre_max = x_max
legendre_scale_factor = 1  # 0e6

s_values = [[] for i in range(len(g))]
s_total = []
vex_values = [[] for i in range(len(g))]
vex_total = []
cave_values = [[] for i in range(len(g))]
cave_total = []
print("processing values")
for x in tqdm(np.linspace(x_min, x_max, increments)):
    for i, (w, b) in enumerate(g):
        s_values[i].append(sigmoid(x, w, b))
        vex_values[i].append(vex_sigmoid(x, w, b))
        cave_values[i].append(cave_sigmoid(x, w, b))
        if not i:
            s_total.append(s_values[i][-1])
            full_vex_legendre_m.append(vex_der(x, w, b))
            vex_total.append(vex_values[i][-1])
            full_cave_legendre_m.append(cave_der(x, w, b))
            cave_total.append(cave_values[i][-1])
        else:
            s_total[-1] += s_values[i][-1]
            full_vex_legendre_m[-1] += vex_der(x, w, b)
            vex_total[-1] += vex_values[i][-1]
            full_cave_legendre_m[-1] += cave_der(x, w, b)
            cave_total[-1] += cave_values[i][-1]

    full_vex_legendre_c.append(vex_total[-1] - (full_vex_legendre_m[-1] * x))
    full_cave_legendre_c.append(cave_total[-1] - (full_cave_legendre_m[-1] * x))
    # if np.random.random() > random_chance:
    # if x < legendre_min or x > legendre_max:
    #     if np.random.random() > random_chance * (x_max - x_min) / (legendre_max - legendre_min):
    # if np.random.random() > random_chance:
    if x not in selected_x:
        del full_vex_legendre_m[-1]
        del full_vex_legendre_c[-1]
        del full_cave_legendre_m[-1]
        del full_cave_legendre_c[-1]

legendre_values_total = []
legendre_values_vex = []
legendre_values_cave = []
legendre_x = np.linspace(x_min, x_max, increments)
print("extracting values from", len(full_vex_legendre_c), "Legendre lines")
for x in tqdm(legendre_x):
    if x >= -0.64:
        print('', end='')
    legendre_values_vex.append(piecewise_value(x, full_vex_legendre_m, full_vex_legendre_c))
    legendre_values_cave.append(piecewise_value(x, full_cave_legendre_m, full_cave_legendre_c, vex=False))
    legendre_values_total.append(
        (legendre_values_vex[-1] + legendre_values_cave[-1]) * (legendre_scale_factor / 2))

linear_collection = []
for m, c in zip(full_vex_legendre_m, full_vex_legendre_c):
    y = []
    for x in np.linspace(x_min, x_max, increments):
        y.append(m * x + c)
    linear_collection.append(y)
for m, c in zip(full_cave_legendre_m, full_cave_legendre_c):
    y = []
    for x in np.linspace(x_min, x_max, increments):
        y.append(m * x + c)
    linear_collection.append(y)

print("plotting")
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.linspace(x_min, x_max, increments)
for line in linear_collection:
    ax1.plot(x, line, color='b', alpha=0.1)

# for s_v, (w, b) in zip(s_values, g):
# ax2.plot(x, s_v, '--', label='(u{:.2f}, s{:.2f})'.format(u, s))
# ax2.plot(x, s_v, '-', alpha=1/np.sqrt(len(g)))
# ax1.grid(None)
# ax1.axis('off')

# ax1.plot(x, vex_total, label='vex_total')
# ax1.plot(x, cave_total, label='cave_total')

ax1.plot(legendre_x, legendre_values_vex, label='legendre_values_vex')
ax1.plot(legendre_x, legendre_values_cave, label='legendre_values_cave')
ax1.plot(legendre_x, legendre_values_total, 'k', label='legendre_values_total')

ax1.plot(x, s_total, 'r--', label='sigmoid_total')

# ax1.set_ylim([-2, len(g) * (narrowness / spread) * 2])
# ax1.set_ylim([-2, len(g) + 1])
ax1.set_ylim([-1, weighting * n_sig * 0.75])
ax2.set_ylim([-1, weighting + 1])
ax1.legend(loc='lower right')

plt.show()

print("done")
