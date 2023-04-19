import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms, datasets
import copy
import matplotlib
matplotlib.use('Agg')
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
num_outputs = 10

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

def piecewise_value(x, net_m, net_c, cavex_m, cavex_c, population, soft=0):
    vexy = []
    cavey = []
    for sm, sc, cvm, cvc in zip(net_m, net_c, cavex_m, cavex_c):
        vexx = torch.matmul(x, torch.transpose(sm + cvm, 0, 1))
        cavex = torch.matmul(x, torch.transpose(sm - cvm, 0, 1))
        vexy.append((vexx + cvc + sc))
        cavey.append((cavex - cvc + sc))
    vexy = torch.stack(vexy)
    cavey = torch.stack(cavey)
    # y_max = torch.max(vexy, dim=2)[0]
    # y_min = torch.min(cavey, dim=2)[0]
    if soft:
        # all_y = []
        # for agent in population:
        #     y_vex = torch.max(vexy[agent.cpu().numpy()], dim=2)[0]
        #     y_cave = torch.max(cavey[agent.cpu().numpy()], dim=2)[0]
        #     all_y.append(torch.transpose((y_vex + y_cave) / 2, 0, 1))
        # return all_y
        soft_y = []
        for agent in population:
            exp_vex = torch.exp(vexy[:, :, agent.cpu().numpy()] / softness)
            soft_vex = torch.stack([ev / torch.sum(exp_vex, dim=2) for ev in torch.movedim(exp_vex, (2, 1, 0), (0, 2, 1))])
            softened_vex = torch.sum(vexy[:, :, agent.cpu().numpy()] * torch.movedim(soft_vex, (0, 2, 1), (2, 1, 0)), dim=2)
            exp_cave = torch.exp(cavey[:, :, agent.cpu().numpy()] / softness)
            soft_cave = torch.stack([ev / torch.sum(exp_cave, dim=2) for ev in torch.movedim(exp_cave, (2, 1, 0), (0, 2, 1))])
            softened_cave = torch.sum(cavey[:, :, agent.cpu().numpy()] * torch.movedim(soft_cave, (0, 2, 1), (2, 1, 0)), dim=2)
            soft_y.append(torch.transpose((softened_vex + softened_cave) / 2, 0, 1))
        return soft_y
    else:
        all_y = []
        for agent in population:
            y_vex = torch.max(vexy[:, :, agent.cpu().numpy()], dim=2)[0]
            y_cave = torch.min(cavey[:, :, agent.cpu().numpy()], dim=2)[0]
            all_y.append(torch.transpose((y_vex + y_cave) / 2, 0, 1))
        return all_y

def iterate_population(population, fitnesses):
    fitnesses = torch.tensor(fitnesses)
    # population = torch.stack(population)
    topk = torch.topk(fitnesses, int(elitism*population_size))
    elite = population[topk[1]]
    children = []
    for c in range(int((1-elitism)*population_size)):
        tournament = np.random.choice(population_size, tournament_size)
        winner = torch.max(fitnesses[tournament], dim=0)[1]
        mutate_indexes = np.array([(np.random.random() < mutation_rate) for i in range(genome_length)])
        child = population[tournament[winner]]
        new_genes = []
        for gene in np.where(mutate_indexes == 1)[0]:
            new_gene = np.random.choice(60000)
            while new_gene in child or new_gene in new_genes:
                new_gene = np.random.choice(60000)
            new_genes.append(new_gene)
        for gene, new_gene in zip(np.where(mutate_indexes == 1)[0], new_genes):
            child[gene] = new_gene
        children.append(child)
    new_population = torch.vstack([elite, torch.stack(children)])
    # [torch.bincount(population.data[i])[torch.bincount(population.data[i]).nonzero()] for i in range(len(population))]
    return new_population

def return_the_best_agent(agent, net_m, net_c, cavex_m, cavex_c, save=0):
    selected_params = {
        'test_accuracy': save,
        'net_m': net_m[:, agent.cpu().numpy(), :],
        'net_c': net_c[:, agent.cpu().numpy()],
        'cavex_m': cavex_m[:, agent.cpu().numpy(), :],
        'cavex_c': cavex_c[:, agent.cpu().numpy()]
    }
    if save:
        torch.save(selected_params, "data/best agent {}".format(test_label))
    return selected_params

# full_sig = []
# full_cavex = []
# full_sig_der = []
# full_cavex_der = []
# full_sig_c = []
# full_cavex_c = []
#
# legendre_scale_factor = 1#0e3
#
# hidden_size = model.layer[0].bias.shape[0]
#
#
# print("extracting Legendre transform")
# for images, labels in tqdm(train_loader):
#     images = images.reshape(-1, 784).to(torch.device(device))
#     for i, (w, b, out_w) in enumerate(zip(model.layer[0].weight.data,
#                                    model.layer[0].bias.data,
#                                    torch.transpose(model.layer[1].weight.data, 0, 1)
#                                    )):
#
#         neuron_sig = Sig(images, w, b, out_w)
#         neuron_cavex = CaVexSig(images, w, out_w)
#         neuron_sig_der = Der(images, w, b, out_w, 'sig')
#         neuron_cavex_der = Der(images, w, b, out_w, False)
#
#         for output, ow in enumerate(out_w):
#             if ow < 0:
#                 neuron_cavex[:, output] *= -1
#                 neuron_cavex_der[:, :, output] *= -1
#
#         if not i:
#             full_sig.append(neuron_sig)
#             full_cavex.append(neuron_cavex)
#             full_sig_der.append(neuron_sig_der)
#             full_cavex_der.append(neuron_cavex_der)
#         else:
#             full_sig[-1] += neuron_sig
#             full_cavex[-1] += neuron_cavex
#             full_sig_der[-1] += neuron_sig_der
#             full_cavex_der[-1] += neuron_cavex_der
#
#     full_sig_c.append(full_sig[-1] - torch.stack(
#         [torch.matmul(im, f) for f, im in zip(full_sig_der[-1], images)]))
#     full_cavex_c.append(full_cavex[-1] - torch.stack(
#         [torch.matmul(im, f) for f, im in zip(full_cavex_der[-1], images)]))
#
# full_sig_c = torch.vstack(full_sig_c)
# full_cavex_c = torch.vstack(full_cavex_c)
# full_sig_der = torch.vstack(full_sig_der)
# full_cavex_der = torch.vstack(full_cavex_der)
#
# torch.save(full_sig_c, 'data/sig_c {}.pt'.format(net_file))
# torch.save(full_cavex_c, 'data/cavex_c {}.pt'.format(net_file))
# torch.save(full_sig_der, 'data/sig_der {}.pt'.format(net_file))
# torch.save(full_cavex_der, 'data/cavex_der {}.pt'.format(net_file))

# full_sig_c = torch.load('data/sig_c {}.pt'.format(net_file))
# full_cavex_c = torch.load('data/cavex_c {}.pt'.format(net_file))
# full_sig_der = torch.load('data/sig_der {}.pt'.format(net_file))
# full_cavex_der = torch.load('data/cavex_der {}.pt'.format(net_file))

soft_levels = [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

generations = 100
population_size = 1000
genome_length = 200  # 1100 <= 200 sigmoid neuron network in terms of CPU/GPU time usage
batch_n = 100
tournament_size = 4
mutation_rate = 0.03
elitism = 0.1
softness = 0.#3

test_label = 'vex_cave g{} p{}x{} t{} m{} e{} s{}'.format(
    generations,
    population_size,
    genome_length,
    tournament_size,
    mutation_rate,
    elitism,
    softness
)

population = torch.stack([torch.tensor(np.random.choice(range(60000), genome_length, replace=False)) for i in range(population_size)])

net_out = []
net_m = []
net_c = []
cavex_out = []
cavex_m = []
cavex_c = []
with torch.no_grad():
    print("calculating Legendre constant")
    # c_list = []
    # for images, labels in tqdm(c_loader):
    #     images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
    #     cavex_const = calculate_c(model, images)
    #     c_list.append(cavex_const)
    # cavex_const = torch.max(torch.stack(c_list), dim=0)[0]
    # cavex_const = torch.ones(num_outputs) * 1
    cavex_const = torch.tensor([
        5.1569, 4.4589, 6.6459, 6.6473, 8.8517, 8.3111, 7.0140, 6.6251, 9.8162, 8.0831])

    print(cavex_const)
    # 21
    # tensor([5.8621, 4.7964, 8.7576, 7.8061, 9.5018, 9.4767, 8.2469, 7.4933,
    #         11.0814, 9.2439])
    # tensor([5.1569, 4.4589, 6.6459, 6.6473, 8.8517, 8.3111, 7.0140, 6.6251, 9.8162,
    #         8.0831])
    # tensor([0.3176, 0.2004, 0.5969, 0.4225, 0.5185, 0.6505, 0.4972, 0.4749, 0.7302,
    #         0.4722])
    # tensor([6.1096, 5.0601, 8.4366, 7.9147, 10.4071, 10.2626, 8.5056, 8.0497,
    #         12.0068, 9.4996])

print("calculating Legendre planes")
for images, labels in tqdm(train_loader):
    images = images.reshape(-1, 784).to(torch.device(device)) - 0.5

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

# net_out = torch.hstack(net_out)
net_m = torch.hstack(net_m)
net_c = torch.hstack(net_c)
# cavex_out = torch.hstack(cavex_out)
cavex_m = torch.hstack(cavex_m)
cavex_c = torch.hstack(cavex_c)


with torch.no_grad():
    generation_metrics = []
    for g in range(generations):
        # correct_m = 0
        correct_l = [0 for i in range(population_size)]
        total = 0
        for batch, (images, labels) in enumerate(tqdm(test_loader)):
            if batch >= batch_n:
                break

            images = images.reshape(-1, 784).to(torch.device(device)) - 0.5
            # out_m = model(images)
            # _, pred = torch.max(out_m, 1)
            # correct_m += (pred == labels).sum().item()

            out_l = piecewise_value(images, net_m, net_c, cavex_m, cavex_c, population, soft=softness)

            for i, out in enumerate(out_l):
                _, pred = torch.max(out, 1)
                correct_l[i] += (pred == labels).sum().item()

            total += labels.size(0)
        if g:
            print("\nGeneration max, min, mean")
            for j, g_met in enumerate(generation_metrics):
                print(j, g_met)
        print(test_label)
        # print('Model testing accuracy: {} %'.format(100 * correct_m / total))
        print('Legendre population testing')
        print("Max = {}   min = {}   average = {}".format(
            np.max(100 * np.array(correct_l) / total),
            np.min(100 * np.array(correct_l) / total),
            np.mean(100 * np.array(correct_l) / total),
        ))
        testing_accuracies = 100 * np.array(correct_l) / total#[100 * np.array(correct_m) / total, 100 * np.array(correct_l) / total]

        # print("Training accuracies:", training_accuracies)
        # print("Testing accuracies:", testing_accuracies)

        generation_metrics.append([
            np.max(100 * np.array(correct_l) / total),
            np.min(100 * np.array(correct_l) / total),
            np.mean(100 * np.array(correct_l) / total)
        ])

        population = iterate_population(population, 100 * np.array(correct_l) / total)
        return_the_best_agent(population[0], net_m, net_c, cavex_m, cavex_c, generation_metrics[-1][0])

        plt.figure()
        x = [i for i in range(len(generation_metrics))]
        g_max = [m for m, _, _ in generation_metrics]
        plt.plot(x, g_max, label='max')
        g_min = [m for _, m, _ in generation_metrics]
        plt.plot(x, g_min, label='min')
        g_ave = [m for _, _, m in generation_metrics]
        plt.plot(x, g_ave, label='mean')
        plt.title("Testing accuracy of GA {}".format(test_label))
        plt.xlabel("Testing accuracy")
        plt.ylabel("Generations")
        plt.legend(loc='lower right')
        figure = plt.gcf()
        figure.set_size_inches(16, 9)
        plt.tight_layout(rect=[0, 0.3, 1, 0.95])
        plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200, format='png')
        plt.close()

print("Done")
