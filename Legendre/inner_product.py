import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms, datasets
from torch.profiler import profile, record_function, ProfilerActivity
import copy
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sn
import pandas as pd
import gc
import itertools
from tqdm import tqdm
from Legendre.train_2D import NeuralNet
from Legendre.train_2D import generate_corner_2class_data, generate_xor_data

sn.set_theme(style="whitegrid")

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
# net_file = 'xor sigmoid nosoftorbias hidden_size[8] test_acc[99.53125]'
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
    # min_cave_min = torch.min(-cavexy, dim=0)[1]
    vex_sy = torch.stack([torch.stack(
        [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(max_vex_max)])
    # cave_sy = torch.stack([torch.stack(
    #     [sy[output_i][j][i] for i, output_i in enumerate(idx)]) for j, idx in enumerate(min_cave_min)])
    # y = (vex_sy + cave_sy) / 2
    return vex_sy

def inner_products(original, cavex):
    # original = torch.nn.functional.normalize(original)
    # cvx_norm = torch.nn.functional.normalize(cavex)
    # org_inv = torch.pinverse(original)
    # cvx_inv = torch.pinverse(cavex)
    # length = len(original)
    # print("inv org")
    # pin_original = torch.stack([torch.pinverse(original[i]) for i in tqdm(range(len(original)))])
    # print("mult org")
    # org_corr = []
    # for i in tqdm(range(length)):
    #     corr_row = []
    #     for j in range(length):
    #         inv_or_I = torch.matmul(pin_original[i], original[j])
    #         difference = torch.sum(torch.square(torch.eye(inv_or_I.shape[0]) - inv_or_I))
    #         corr_row.append(difference)
    #     print(torch.topk(-torch.stack(corr_row), 10)[0])
    #     org_corr.append(torch.stack(corr_row))

    length = len(cavex)
    print("inv org")
    pin_cavex = torch.stack([torch.pinverse(cavex[i]) for i in tqdm(range(len(cavex)))])
    print("mult org")
    org_corr = []
    for i in tqdm(range(length)):
        corr_row = []
        for j in range(length):
            inv_or_I = torch.matmul(pin_cavex[i], cavex[j])
            difference = torch.sum(torch.square(torch.eye(inv_or_I.shape[0]) - inv_or_I))
            corr_row.append(difference)
        print(torch.topk(-torch.stack(corr_row), 10)[0])
        org_corr.append(torch.stack(corr_row))

    # org_corr = torch.stack([torch.stack([
    #     torch.matmul(pin_original[i], original[j]) for j in range(length)]) for i in range(length)])
    print("inv cavex")
    pin_cavex = torch.stack([torch.pinverse(cavex[i]) for i in range(len(cavex))])
    print("mult cavex")
    cavex_corr = torch.stack([torch.stack([
        torch.matmul(pin_cavex[i], cavex[j]) for j in range(length)]) for i in range(length)])

    # org_corr = org_inv * org_inv
    # cvx_corr = cvx_inv * cvx_inv
    # org_corr = org_norm * org_norm
    # cvx_corr = cvx_norm * cvx_norm

    fig, axs = plt.subplots(2, 1)
    length = len(original)
    original_df = pd.DataFrame(org_corr, range(len(org_corr)), range(len(org_corr[0])))
    axs[0] = sn.heatmap(original_df, annot=True, annot_kws={'size': 8}, ax=axs[0])
    axs[0].set_title("Original ")
    cavex_df = pd.DataFrame(cavex_corr, range(len(cavex_corr)), range(len(cavex_corr[0])))
    axs[1] = sn.heatmap(cavex_df, annot=True, annot_kws={'size': 8}, ax=axs[1])
    axs[1].set_title("Testing confusion")
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle("inner product {}.png".format(net_file), fontsize=16)
    plt.savefig("./plots/inner product {}.png".format(net_file), bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()

full_sig_c = torch.load('data/sig_c {}.pt'.format(net_file))
full_cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))
full_sig_der = torch.load('data/sig_der {}.pt'.format(net_file))
full_cavex_der = torch.load('data/cavex_der {}.pt'.format(net_file))

inner_products(full_sig_der, full_cavex_der)

with torch.no_grad():
    correct_m = 0
    total = 0
    repeats = 5
    number_of_legendre = len(full_sig_der)
    for repeat in range(repeats):
        for images, labels in tqdm(test_loader):
            images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

            print("Profiling the neural network")

            with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    out_m = model(images)

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            # # Enable profiling
            # with torch.profiler.profile(profile_memory=True, record_shapes=True) as prof:
            #     # Run the model with the dummy input
            #     out_m = model(images)
            #
            # # Print the CPU profiling results
            # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
            # # Print the GPU profiling results
            # print(prof.key_averages(group_by_input_shape=True))
            # # Print the profiling results
            # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            break

    sample_intervals = 600
    scores = {}
    for l_amount in np.linspace(1, number_of_legendre, sample_intervals, dtype=int):
        print("starting Legendre sample amount", l_amount)
        legendre_sample = np.random.choice(range(number_of_legendre), l_amount, replace=False)

        a, b, c, d = full_sig_der[legendre_sample], \
                     full_sig_c[legendre_sample], \
                     full_cavex_der[legendre_sample], \
                     full_cavex_c[legendre_sample]
        for repeat in range(repeats):
            for images, labels in tqdm(test_loader):
                images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

                print("Profiling Legendre", l_amount)

                torch.cuda.empty_cache()

                with profile(activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("model_inference"):
                        out_l = piecewise_value(images, a, b, c, d)

                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                break


print("Done")

