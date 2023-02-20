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

def gaussian(x, mean, std, w):
    return w * np.exp(-np.square(mean - x)/(2*np.square(std)))

def vex_gaussian(x, mean, std, w):
    c = 0.5 / (np.square(std))
    return w * np.exp(-np.square(mean - x)/(2*np.square(std))) + c*(x**2)

def vex_der(x, mean, std, w):
    c = 0.5 / (np.square(std))
    return w * ((mean - x) / np.square(std)) * np.exp(-np.square(mean - x)/(2*np.square(std))) + (2*c*x)

def cave_gaussian(x, mean, std, w):
    c = 0.5 / (np.square(std))
    return w * np.exp(-np.square(mean - x)/(2*np.square(std))) - c*(x**2)

def cave_der(x, mean, std, w):
    c = 0.5 / (np.square(std))
    return w * ((mean - x) / np.square(std)) * np.exp(-np.square(mean - x)/(2*np.square(std))) - (2*c*x)

def piecewise_value(x, legendre_m, legendre_c, vex=True):
    y = []
    for m, c in zip(legendre_m, legendre_c):
        y.append((m*x + c) / legendre_scale_factor)
    if vex:
        return np.max(y)
    else:
        return np.min(y)

# g = [
#     [0, 1],
#     [1, 0.9]
# ]

spread = 6
narrowness = 0.2
weighting = 2
n_g = 2000
g = [[np.random.random()*spread - (spread/2),
      np.random.random()*narrowness,
      np.random.random() * weighting - (weighting / 2)]
     for i in range(n_g)]

x_min = -3.5
x_max = 3.5
increments = 600

full_vex_legendre_m = []
full_cave_legendre_m = []
full_vex_legendre_c = []
full_cave_legendre_c = []
random_chance = 0.3
l_amount = 269
selected_x = np.linspace(x_min, x_max, increments)[np.linspace(0, increments-1, l_amount, dtype=int)]
legendre_min = -(spread / 2) - (3*narrowness)
legendre_max = (spread / 2) + (3*narrowness)
legendre_scale_factor = 10e6

g_values = [[] for i in range(len(g))]
g_total = []
vex_values = [[] for i in range(len(g))]
vex_total = []
cave_values = [[] for i in range(len(g))]
cave_total = []
print("processing values")
for x in tqdm(np.linspace(x_min, x_max, increments)):
    for i, (u, s, w) in enumerate(g):
        g_values[i].append(gaussian(x, u, s, w))
        vex_values[i].append(vex_gaussian(x, u, s, w))
        cave_values[i].append(cave_gaussian(x, u, s, w))
        if not i:
            g_total.append(g_values[i][-1])
            full_vex_legendre_m.append(vex_der(x, u, s, w))
            vex_total.append(vex_values[i][-1])
            full_cave_legendre_m.append(cave_der(x, u, s, w))
            cave_total.append(cave_values[i][-1])
        else:
            g_total[-1] += g_values[i][-1]
            full_vex_legendre_m[-1] += vex_der(x, u, s, w)
            vex_total[-1] += vex_values[i][-1]
            full_cave_legendre_m[-1] += cave_der(x, u, s, w)
            cave_total[-1] += cave_values[i][-1]

    full_vex_legendre_c.append(vex_total[-1] - (full_vex_legendre_m[-1] * x))
    full_cave_legendre_c.append(cave_total[-1] - (full_cave_legendre_m[-1] * x))
    # if np.random.random() > random_chance:
    # if x < legendre_min or x > legendre_max:
    #     if np.random.random() > random_chance * (x_max - x_min) / (legendre_max - legendre_min):
    if x not in selected_x:
        del full_vex_legendre_m[-1]
        del full_vex_legendre_c[-1]
        del full_cave_legendre_m[-1]
        del full_cave_legendre_c[-1]

legendre_values_total = []
legendre_values_vex = []
legendre_values_cave = []
legendre_x = np.linspace(x_min, x_max, increments*3)
print("extracting values from", len(full_vex_legendre_c), "Legendre lines")
for x in tqdm(legendre_x):
    legendre_values_vex.append(piecewise_value(x, full_vex_legendre_m, full_vex_legendre_c))
    legendre_values_cave.append(piecewise_value(x, full_cave_legendre_m, full_cave_legendre_c, vex=False))
    legendre_values_total.append(
        (legendre_values_vex[-1] + legendre_values_cave[-1]) * (legendre_scale_factor / 2))

linear_collection = []
for m, c in zip(full_vex_legendre_m, full_vex_legendre_c):
    y = []
    for x in np.linspace(x_min, x_max, increments):
        y.append(m*x + c)
    linear_collection.append(y)
for m, c in zip(full_cave_legendre_m, full_cave_legendre_c):
    y = []
    for x in np.linspace(x_min, x_max, increments):
        y.append(m*x + c)
    linear_collection.append(y)

print("plotting")
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.linspace(x_min, x_max, increments)
# for line in linear_collection:
#     ax1.plot(x, line, color='r', alpha=0.1)

for g_v, (u, s, w) in zip(g_values, g):
    # ax2.plot(x, g_v, '--', label='(u{:.2f}, s{:.2f})'.format(u, s))
    ax2.plot(x, g_v, '-', alpha=1/np.sqrt(len(g)))
ax2.grid(None)
ax2.axis('off')

# ax1.plot(x, vex_total, label='vex_total')
# ax1.plot(x, cave_total, label='cave_total')

# ax1.plot(legendre_x, legendre_values_vex, label='legendre_values_vex')
# ax1.plot(legendre_x, legendre_values_cave, label='legendre_values_cave')
ax1.plot(legendre_x, legendre_values_total, 'k', label='legendre_values_total')

ax1.plot(x, g_total, 'r--', label='gaussian_total')

# ax1.set_ylim([-2, len(g) * (narrowness / spread) * 2])
ax1.set_ylim([-15, 15])
ax2.set_ylim([0, 2])
ax1.legend(loc='lower right')

plt.show()

print("done")


