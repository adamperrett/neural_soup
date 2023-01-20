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
sns.set_theme(style="whitegrid")

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

input_size = 28 * 28
num_classes = 10
batch_size = 128
num_epochs = 100
lr = 0.03
momentum = 0.9

hidden_size = [512, 512, 512]

in_out_neurons = 10
evo_rate = 10
rate_reduction = 1.05

# levels_of_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
levels_of_dropout = [0.5]
neuron_types = [['relu'],
                ['tanh'],
                ['sig'],
                ['smin'],
                ['smax'],
                ['smin', 'smax'],
                ['gelu'],
                ['lrelu'],
                ['relu', 'tanh', 'sig'],
                ['relu', 'gelu', 'lrelu'],
                ['relu', 'tanh', 'sig', 'gelu', 'lrelu'],
                ['relu', 'tanh', 'sig', 'smin', 'smax', 'gelu', 'lrelu']]
colours = pl.cm.gist_rainbow(np.linspace(0, 1, len(neuron_types)))

test_label = "evo_rate{}x{}x{} drop{} hidden_size{} lr{} bs{}".format(
    in_out_neurons,
    evo_rate,
    rate_reduction,
    levels_of_dropout[0],
    hidden_size,
    lr,
    batch_size)

trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, generator=torch.Generator(device=device))


class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mask = binomial.sample(x.size())
            return x * mask * (1.0 / (1 - self.p)), mask
        return x, None


class MyLinear(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, bias=True, drop_input=True):
        super(MyLinear, self).__init__(in_feats, out_feats, bias=bias)
        self.drop_input = drop_input
        self.custom_dropout = MyDropout(p=drop_p)

    def forward(self, input):
        if self.drop_input:
            dropout_value, mask = self.custom_dropout(input)
            return F.linear(dropout_value, self.weight, self.bias), mask
        else:
            dropout_value, mask = self.custom_dropout(self.weight)
            return F.linear(input, dropout_value, self.bias), mask


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p, neuron_types, dropout=True):
        super(NeuralNet, self).__init__()

        self.dropout = dropout
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(input_size, hidden_size[0]))
        self.neuron_types = neuron_types
        self.splits = [np.random.choice(self.neuron_types, hidden_size[0])]

        for i in range(len(hidden_size)-1):
            if dropout:
                self.layer.append(MyLinear(hidden_size[i], hidden_size[i+1], p))
            else:
                self.layer.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.splits.append(np.random.choice(self.neuron_types, hidden_size[i+1]))

        self.split_to_idx()

        if dropout:
            self.layer.append(MyLinear(hidden_size[-1], num_classes, p))
        else:
            self.layer.append(nn.Linear(hidden_size[-1], num_classes))

        self.functions = {'relu': nn.ReLU(),
                          'tanh': nn.Tanh(),
                          'sig': nn.Sigmoid(),
                          'smin': nn.Softmax(dim=1),
                          'smax': nn.Softmin(dim=1),
                          'gelu': nn.GELU(),
                          # 'mha': nn.MultiheadAttention(),
                          'lrelu': nn.LeakyReLU()
                          }

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def split_to_idx(self):
        self.act_idxs = []
        for split in self.splits:
            self.act_idxs.append({n_type: np.where(split == n_type)[0] for n_type in self.neuron_types})

    def mixed_act(self, x, layer):
        combined = torch.zeros([len(x), len(self.splits[layer])])
        for n_type in self.neuron_types:
            combined[:, self.act_idxs[layer][n_type]] = self.functions[n_type](x[:, self.act_idxs[layer][n_type]])
        return combined

    def forward(self, x):
        masks = []
        out = self.layer[0](x)
        out = self.mixed_act(out, 0)
        for i in range(1, len(self.layer) - 1):
            if self.dropout:
                out, mask = self.layer[i](out)
                masks.append(mask)
            else:
                out = self.layer[i](out)
            out = self.mixed_act(out, i)
        if self.dropout:
            out, mask = self.layer[-1](out)
            masks.append(mask)
        else:
            out = self.layer[-1](out)
        out = self.LogSoftmax(out)
        return out, masks

def process_mask_and_loss(mask_and_loss):
    network = []
    for masks, loss in mask_and_loss:
        layers = []
        for mask in masks:
            weighted_mask = torch.mul(loss.reshape([loss.shape[0], 1]), mask)
            layers.append(torch.sum(weighted_mask, dim=0) / torch.sum(mask, dim=0))
        network.append(layers)
    # print("new batch")
    # for n, nt in zip(network, neuron_types):
    #     print("min", torch.min(n[0]), "max", torch.max(n[0]), "diff", torch.max(n[0]) - torch.min(n[0]), nt)
    return network

def add_remove_neurons(weighted_masks):
    with torch.no_grad():
        for setting, m in zip(weighted_masks, models):
            for mask, layer, split in zip(setting, models[m].layer[:-1], models[m].splits):
                # mask = torch.nan_to_num(mask, nan=torch.max(mask))
                min_k_v, min_k_i = torch.topk(mask, in_out_neurons, largest=False)
                max_k_v, max_k_i = torch.topk(mask, in_out_neurons)
                # add mutation and crossover here
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                layer.weight[max_k_i] = layer.weight[min_k_i]
                layer.bias[max_k_i] = layer.bias[min_k_i]
                split[max_k_i.to('cpu')] = split[min_k_i.to('cpu')]
                layer.bias.requires_grad = True
                layer.weight.requires_grad = True
                # reset the weight momentum values
            models[m].split_to_idx()
        # print("Done shifting")


models = {}
params = []
# for p in levels_of_dropout:
for nt in neuron_types:
    models['{}'.format(nt)] = NeuralNet(input_size, hidden_size, num_classes,
                                       neuron_types=nt, p=levels_of_dropout[0]).to(device)
    params.append({'params': models['{}'.format(nt)].parameters()})

lossFunction = nn.NLLLoss(reduction='none')
optimize_all = optim.SGD(params,
                         lr=lr, momentum=momentum)

training_losses = []
testing_accuracies = []

for epoch in range(num_epochs):
    print(test_label)
    processed_masks = []
    for p in models:
        models[p].train()
    loss_ = [0 for i in range(len(neuron_types))]
    for batch, (images, labels) in enumerate(train_loader):
        # Flatten the input images of [28,28] to [1,784]
        images = images.reshape(-1, 784).to(torch.device(device))

        output = []
        for p in models:
            output.append(models[p](images))

        loss = []
        mask_and_example_loss = []
        for out, masks in output:
            loss_each = lossFunction(out, labels)
            mask_and_example_loss.append([masks, loss_each])
            loss.append(torch.mean(loss_each))

        if batch % evo_rate == evo_rate - 1:
            processed_mask = process_mask_and_loss(mask_and_example_loss)
            #
            # testing_accuracy_before = []
            # with torch.no_grad():
            #     for p in models:
            #         models[p].eval()
            #     correct = [0 for i in range(len(neuron_types))]
            #     total = 0
            #     for images, labels in test_loader:
            #         images = images.reshape(-1, 784).to(torch.device(device))
            #         out = []
            #         for p in models:
            #             out.append(models[p](images))
            #         predicted = []
            #         for o, _ in out:
            #             _, pred = torch.max(o, 1)
            #             predicted.append(pred)
            #         for i in range(len(neuron_types)):
            #             correct[i] += (predicted[i] == labels).sum().item()
            #         total += labels.size(0)
            #     for i in range(len(neuron_types)):
            #         print('Testing accuracy: {} %  {}'.format(100 * correct[i] / total, neuron_types[i]))
            #         testing_accuracy_before.append(100 * np.array(correct[i]) / total)

            add_remove_neurons(processed_mask)

            # testing_accuracy_after = []
            # with torch.no_grad():
            #     for p in models:
            #         models[p].eval()
            #     correct = [0 for i in range(len(neuron_types))]
            #     total = 0
            #     for images, labels in test_loader:
            #         images = images.reshape(-1, 784).to(torch.device(device))
            #         out = []
            #         for p in models:
            #             out.append(models[p](images))
            #         predicted = []
            #         for o, _ in out:
            #             _, pred = torch.max(o, 1)
            #             predicted.append(pred)
            #         for i in range(len(neuron_types)):
            #             correct[i] += (predicted[i] == labels).sum().item()
            #         total += labels.size(0)
            #     for i in range(len(neuron_types)):
            #         print('Testing accuracy: {} %  {}'.format(100 * correct[i] / total, neuron_types[i]))
            #         testing_accuracy_after.append(100 * np.array(correct[i]) / total)
            # for i in range(len(neuron_types)):
            #     print('Testing accuracy difference: {} %  {}'.format(
            #         testing_accuracy_after[i] - testing_accuracy_before[i],
            #         neuron_types[i]))
            # for p in models:
            #     models[p].train()

        # processed_masks.append(processed_mask)
        else:
            optimize_all.zero_grad()

            for l in loss:
                l.backward()

            optimize_all.step()

        for i in range(len(neuron_types)):
            loss_[i] += loss[i]

    for i in range(len(neuron_types)):
        print("Epoch{}, Training loss:{} types:{}".format(epoch,
                                                             loss_[i] / len(train_loader),
                                                             neuron_types[i]))
    training_losses.append(loss_)

    # Testing
    with torch.no_grad():
        for p in models:
            models[p].eval()
        correct = [0 for i in range(len(neuron_types))]
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 784).to(torch.device(device))
            out = []
            for p in models:
                out.append(models[p](images))
            predicted = []
            for o, _ in out:
                _, pred = torch.max(o, 1)
                predicted.append(pred)
            for i in range(len(neuron_types)):
                correct[i] += (predicted[i] == labels).sum().item()
            total += labels.size(0)
        for i in range(len(neuron_types)):
            print('Testing accuracy: {} %  {}'.format(100 * correct[i] / total, neuron_types[i]))
        testing_accuracies.append(100 * np.array(correct) / total)

    if len(testing_accuracies) % 10 == 0:
        print("plotting")
        plt.figure()
        for i, p, in enumerate(models):
            print("\n", p, "\n", np.array(testing_accuracies).astype(float)[:, i])
            plt.plot([x for x in range(len(np.array(testing_accuracies).astype(float)[:, i]))],
                     np.array(testing_accuracies).astype(float)[:, i], label=p, color=colours[i])
        plt.ylim([85, 100])
        plt.xlabel('epoch')
        plt.ylabel('test accuracy')
        plt.legend(loc='lower right')
        figure = plt.gcf()
        figure.set_size_inches(16, 9)
        plt.tight_layout(rect=[0, 0.3, 1, 0.95])
        plt.suptitle(test_label, fontsize=16)
        plt.grid(visible=None, which='both')
        plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200, format='png')
        plt.close()

    evo_rate *= rate_reduction

# torch.save(model, 'mnist_model.pt')
print("training:")
for i, p, in enumerate(models):
    print(p, np.array(training_losses).astype(float)[:, i])
# print(training_losses)
print("testing")

plt.figure()
for i, p, in enumerate(models):
    print("\n", p, "\n", np.array(testing_accuracies).astype(float)[:, i])
    plt.plot([x for x in range(len(np.array(testing_accuracies).astype(float)[:, i]))],
             np.array(testing_accuracies).astype(float)[:, i], label=p, color=colours[i])
plt.ylim([95, 100])
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.legend(loc='lower right')
figure = plt.gcf()
figure.set_size_inches(16, 9)
plt.tight_layout(rect=[0, 0.3, 1, 0.95])
plt.suptitle(test_label, fontsize=16)
plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200, format='png')
plt.close()
# print(testing_accuracies)

print('done')