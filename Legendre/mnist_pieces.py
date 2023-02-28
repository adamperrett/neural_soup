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

def piecewise_value(x, sig_m, sig_c, cavex_m, cavex_c, soft=False):
    sy = []
    cavexy = []
    for sm, sc, cvm, cvc in zip(sig_m, sig_c, cavex_m, cavex_c):
        smx = torch.matmul(x, sm)
        cavexx = torch.matmul(x, cvm)
        sy.append(smx + sc)
        cavexy.append(cavexx + cvc)
    sy = torch.stack(sy)
    cavexy = torch.stack(cavexy)
    max_vex_max = torch.max(cavexy, dim=0)[1]
    min_cave_min = torch.min(-cavexy, dim=0)[1]
    vex_sy = torch.stack([torch.stack(
        [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max)])
    cave_sy = torch.stack([torch.stack(
        [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min)])
    y = (vex_sy + cave_sy) / 2
    return y

full_sig_c = torch.load('data/sig_c {}.pt'.format(net_file))
full_cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))
full_sig_der = torch.load('data/sig_der {}.pt'.format(net_file))
full_cavex_der = torch.load('data/cavex_der {}.pt'.format(net_file))

with torch.no_grad():
    correct_m = 0
    total = 0
    number_of_legendre = len(full_sig_der)
    for images, labels in tqdm(test_loader):
        images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
        out_m = model(images)
        _, pred = torch.max(out_m, 1)
        correct_m += (pred == labels).sum().item()
        total += labels.size(0)
    print('Model testing accuracy: {} %'.format(100 * correct_m / total))
    model_test_accuracy = 100 * correct_m / total

    sample_intervals = 100
    repeats = 10
    scores = {}
    for l_amount in np.linspace(1, number_of_legendre, sample_intervals, dtype=int):
        print("starting Legendre sample amount", l_amount)
        amount_sample = []
        for repeat in range(repeats):
            correct_l = 0
            total = 0
            legendre_sample = np.random.choice(range(number_of_legendre), l_amount, replace=False)
            for images, labels in tqdm(test_loader):
                images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
                out_l = piecewise_value(images,
                                        full_sig_der[legendre_sample],
                                        full_sig_c[legendre_sample],
                                        full_cavex_der[legendre_sample],
                                        full_cavex_c[legendre_sample])
                _, pred = torch.max(out_l, 1)
                correct_l += (pred == labels).sum().item()
                total += labels.size(0)
            test_accuracy = 100 * correct_l / total
            print('Legendre testing accuracy: {} % for L_amount{} repeat{}'.format(
                test_accuracy, l_amount, repeat))
            amount_sample.append(test_accuracy)
        scores['{}'.format(l_amount)] = amount_sample
        for setting in scores:
            print("Full scores for {} = {}".format(setting, scores[setting]))
        print("Model test accuracy {}%".format(model_test_accuracy))
        for setting in scores:
            print("For amount", setting, "mean = {:.2F}    stdev = {:.2F}    min = {:.2F}    max = {:.2F}".format(
                np.mean(scores[setting]),
                np.std(scores[setting]),
                np.min(scores[setting]),
                np.max(scores[setting])))

print("Done")

