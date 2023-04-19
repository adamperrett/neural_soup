import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms, datasets
import copy
import matplotlib.pyplot as plt

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            return X * binomial.sample(X.size()) * (1.0 / (1 - self.p))
        return X


class MyLinear(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, bias=True, drop_input=True):
        super(MyLinear, self).__init__(in_feats, out_feats, bias=bias)
        self.drop_input = drop_input
        self.custom_dropout = MyDropout(p=drop_p)

    def forward(self, input):
        if self.drop_input:
            dropout_value = self.custom_dropout(input)
            return F.linear(dropout_value, self.weight, self.bias)
        else:
            dropout_value = self.custom_dropout(self.weight)
            return F.linear(input, dropout_value, self.bias)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p, neuron_types, dropout=False):
        super(NeuralNet, self).__init__()
        # if dropout:
        #     self.fc1 = MyLinear(input_size, hidden_size, p)
        # else:
        self.layer = nn.ModuleList()
        self.layer.append(nn.Linear(input_size, hidden_size[0]))
        self.neuron_types = neuron_types
        self.splits = [np.random.choice(self.neuron_types, hidden_size[0])]
        self.act_idxs = [{n_type: np.where(self.splits[-1] == n_type)[0] for n_type in self.neuron_types}]

        for i in range(len(hidden_size) - 1):
            self.layer.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.splits.append(np.random.choice(self.neuron_types, hidden_size[i + 1]))
            self.act_idxs.append({n_type: np.where(self.splits[-1] == n_type)[0] for n_type in self.neuron_types})

        self.layer.append(nn.Linear(hidden_size[-1], num_classes, bias=False))

        self.functions = {'relu': nn.ReLU(),
                          'tanh': nn.Tanh(),
                          'sig': nn.Sigmoid(),
                          'smin': nn.Softmax(dim=1),
                          'smax': nn.Softmin(dim=1),
                          'gelu': nn.GELU(),
                          # 'mha': nn.MultiheadAttention(),
                          'lrelu': nn.LeakyReLU()
                          }

        # self.fc2 = nn.Linear(hidden_size, num_classes)
        # if dropout:
        #     self.fc2 = MyLinear(hidden_size, num_classes, p)
        # else:
        #     self.fc2 = nn.Linear(hidden_size, num_classes)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def mixed_act(self, x, layer):
        combined = torch.zeros([len(x), len(self.splits[layer])]).type(torch.float32)
        for n_type in self.neuron_types:
            combined[:, self.act_idxs[layer][n_type]] = self.functions[n_type](x[:, self.act_idxs[layer][n_type]])
        return combined

    def forward(self, x):
        out = self.layer[0](x)
        out = self.mixed_act(out, 0)
        for i in range(1, len(self.layer) - 1):
            out = self.layer[i](out)
            out = self.mixed_act(out, i)
        out = self.layer[-1](out)
        # out = self.LogSoftmax(out)
        return out

    def separate_outputs(self, output):
        def output_n(x):
            out = self.layer[0](x)
            out = self.mixed_act(out, 0)
            for i in range(1, len(self.layer) - 1):
                out = self.layer[i](out)
                out = self.mixed_act(out, i)
            out = self.layer[-1](out)
            # out = self.LogSoftmax(out)
            return out[:, output]
        return output_n


def check_memory(where=''):
    print(where)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print("   total     reserved     allocated     free")
    print(["{0:.2E}".format(thing) for thing in [t, r, a, f]])

def generate_xor_data(batch_size=32, batches=40):
    data = []
    for i in range(batches):
        batch_d = []
        batch_l = []
        for b in range(batch_size):
            x = torch.rand(1) * 2 - 1
            y = torch.rand(1) * 2 - 1
            batch_d.append(torch.tensor([x, y]))
            batch_l.append(torch.Tensor((x * y) < 0).type(torch.int64))
        data.append([torch.vstack(batch_d), torch.vstack(batch_l).squeeze()])
    return data

def generate_corner_2class_data(batch_size=32, batches=40):
    data = []
    for i in range(batches):
        batch_d = []
        batch_l = []
        for b in range(batch_size):
            x = torch.rand(1) * 2 - 1
            y = torch.rand(1) * 2 - 1
            # x = torch.round(torch.rand(1) * 3 - 1.5).type(torch.int).type(torch.float32)
            # y = torch.round(torch.rand(1) * 3 - 1.5).type(torch.int).type(torch.float32)
            # x = torch.ones(1) * -1
            # y = torch.ones(1) * -1
            # x = torch.zeros(1)
            # y = torch.zeros(1)
            batch_d.append(torch.tensor([x, y]))
            batch_l.append(torch.Tensor(x < 0 and y < 0).type(torch.int64))
        data.append([torch.vstack(batch_d), torch.vstack(batch_l).squeeze()])
    return data


if __name__ == '__main__':

    torch.manual_seed(272727)

    input_size = 2
    num_classes = 2
    batch_size = 64
    num_epochs = 500
    lr = 0.008
    momentum = 0.9

    hidden_size = [8]

    # levels_of_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    levels_of_dropout = [0.5]
    neuron_types = [
        # ['relu'],
        # ['tanh'],
        ['sig'],
        # ['smin'],
        # ['smax'],
        # ['smin', 'smax'],
        # ['gelu'],
        # ['lrelu'],
    ]

    models = {}
    params = []
    # for p in levels_of_dropout:
    for nt in neuron_types:
        models['{}'.format(nt)] = NeuralNet(input_size, hidden_size, num_classes,
                                            neuron_types=nt, p=levels_of_dropout[0]).to(device)
        params.append({'params': models['{}'.format(nt)].parameters()})

    lossFunction = nn.CrossEntropyLoss()
    optimize_all = optim.SGD(params,
                             lr=lr, momentum=momentum)

    # train_loader = generate_corner_2class_data(batches=40, batch_size=batch_size)
    # test_loader = generate_corner_2class_data(batches=10, batch_size=batch_size)
    train_loader = generate_xor_data(batches=40, batch_size=batch_size)
    test_loader = generate_xor_data(batches=10, batch_size=batch_size)

    training_losses = []
    testing_accuracies = []

    for epoch in range(num_epochs):
        loss_ = [0 for i in range(len(neuron_types))]
        for images, labels in train_loader:

            output = []
            for p in models:
                output.append(models[p](images))

            loss = []
            for out in output:
                loss.append(lossFunction(out, labels))

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
            correct = [0 for i in range(len(neuron_types))]
            total = 0
            for images, labels in test_loader:
                out = []
                for p in models:
                    out.append(models[p](images))
                predicted = []
                for o in out:
                    _, pred = torch.max(o, 1)
                    predicted.append(pred)
                for i in range(len(neuron_types)):
                    correct[i] += (predicted[i] == labels).sum().item()
                total += labels.size(0)
            for i in range(len(neuron_types)):
                print('Testing accuracy: {} %  {}'.format(100 * correct[i] / total, neuron_types[i]))
            testing_accuracies.append(100 * np.array(correct) / total)

    print("training:")
    for i, p, in enumerate(models):
        print(p, np.array(training_losses).astype(float)[:, i])
    # print(training_losses)
    print("testing")
    for i, p, in enumerate(models):
        print("\n", p, "\n", np.array(testing_accuracies).astype(float)[:, i])

    test_label = "hidden_size{} test_acc{}".format(hidden_size, testing_accuracies[-1])
    for m in models:
        # torch.save(models[m], 'data/xor relu nosoftorbias {}.pt'.format(test_label))
        # torch.save(models[m], 'xor sigmoid nosoftorbias {}.pt'.format(test_label))
        torch.save(models[m], 'data/xor sigmoid separate_out {}.pt'.format(test_label))
        # torch.save(models[m], 'corner sigmoid nosoftorbias {}.pt'.format(test_label))

    print('done')
