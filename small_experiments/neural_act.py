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
sns.set_theme(style="whitegrid")

seed = 272727
torch.manual_seed(seed)
np.random.seed(seed)

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

'''
Train an adaptive neuron. It starts as a triangle kernel and learns 'broader' representations.
'''

input_size = 28 * 28
num_classes = 10
num_epochs = 100
lr = 0.0003
momentum = 0.0000009
learning = 1

retest_rate = 10

error_threshold = 0.3
node = 0
tri = 1
'''
best widths
n0t1 = 0.1 
n1t1 = 0.075
n0t0 = 0.1
n1t0 = 0.025 but lots of n 0.05 also ok
'''

# parameter_settings = [0.075, 0.1, 0.125, 0.15]
# parameter_settings = [0.04]
# parameter_settings = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3]
parameter_settings = [0.1]
colours = pl.cm.gist_rainbow(np.linspace(0, 1, len(parameter_settings)))


test_label = "2n{}t{} w{} et{} learning{}".format(node, tri, parameter_settings, error_threshold, learning)

batch_size_train = 32
batch_size_test = 32

trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                           shuffle=True, generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                          shuffle=True, generator=torch.Generator(device=device))


class Triangle(nn.Module):
    def __init__(self, mean, std):
        super(Triangle, self).__init__()
        # self.mean = torch.nn.Parameter(mean)
        self.mean = mean
        self.std = torch.nn.Parameter(torch.Tensor([std for i in range(len(mean))]))
        # self.std = torch.nn.Parameter(std)
        # self.std = std

    def forward(self, x, batch_size):
        broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # tout = torch.exp((-(torch.sum(torch.square(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std.unsqueeze(1) ** 2))
        tout = torch.max(torch.tensor(0), 1 - torch.transpose(torch.abs(broadcast - x), 0, 1) / self.std.unsqueeze(1))
        # tout = torch.max(torch.tensor(0), 1 - torch.abs(broadcast - x) / self.std)
        # tout = torch.exp((-(torch.sum(torch.square(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std ** 2))
        # tout = torch.exp((-(torch.sum(torch.abs(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std ** 2))
        return torch.transpose(tout, 0, 1)

class Gaussian(nn.Module):

    def __init__(self, mean=0, std=1):
        super(Gaussian, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x, batch_size):
        broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # broadcast = torch.stack([self.mean for i in range(len(x))])
        gauss = torch.exp((-(broadcast - x) ** 2)/(2 * self.std ** 2))
        # return torch.clamp(gauss, min=self.min, max=self.max)
        return gauss

class Node_Triangle(nn.Module):
    def __init__(self, mean, std):
        super(Node_Triangle, self).__init__()
        # self.mean = torch.nn.Parameter(mean)
        self.mean = mean
        # self.std = torch.nn.Parameter(torch.Tensor([std for i in range(len(mean))]))
        self.std = std

    def forward(self, x, batch_size):
        broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # tout = torch.exp((-(torch.sum(torch.square(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std.unsqueeze(1) ** 2))
        tout = torch.max(torch.tensor(0), 1 - (torch.sum(torch.square(broadcast - x), dim=2) / input_size) / self.std)
        # tout = torch.exp((-(torch.sum(torch.square(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std ** 2))
        # tout = torch.exp((-(torch.sum(torch.abs(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std ** 2))
        return tout

class Node_Gaussian(nn.Module):

    def __init__(self, mean=0, std=1):
        super(Node_Gaussian, self).__init__()
        self.mean = torch.nn.Parameter(mean)
        # self.mean = mean
        self.std = torch.nn.Parameter(torch.Tensor([std for i in range(len(mean))]))
        # self.std = std

    def forward(self, x, batch_size):
        broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        # broadcast = torch.transpose(torch.stack([self.mean for i in range(batch_size)]), 0, 1)
        gauss = torch.exp((-(torch.sum(torch.square(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std.unsqueeze(1) ** 2))
        # gauss = torch.exp((-(torch.sum(torch.square(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std ** 2))
        # gauss = torch.exp((-(torch.sum(torch.abs(broadcast - x), dim=2) / input_size) ** 2)/(2 * self.std ** 2))
        return gauss

class Average_with_mask(nn.Module):
    def __init__(self, mask):
        super(Average_with_mask, self).__init__()
        self.mask = mask

    def forward(self, x):
        denom = torch.sum(self.mask, -1, keepdim=True)
        out = torch.sum(torch.transpose(x, 0, 1) * self.mask, dim=2) / torch.transpose(denom, 0, 1)
        return torch.transpose(out, 0, 1)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, num_classes, centres, synapse_mask, output_errors, stdev, node=True, tri=True):
        super(NeuralNet, self).__init__()
        self.hidden_size = len(centres)
        self.synapse_mask = synapse_mask
        self.node = node

        if node:
            if tri:
                self.act_hidden = Node_Triangle(mean=centres, std=stdev)
            else:
                self.act_hidden = Node_Gaussian(mean=centres, std=stdev)
        else:
            if tri:
                self.act_hidden = Triangle(mean=centres, std=stdev)
            else:
                self.act_hidden = Gaussian(mean=centres, std=stdev)
            self.average_synapses = Average_with_mask(synapse_mask)
            self.act_out = Gaussian(mean=torch.ones(self.hidden_size), std=stdev)
        self.output_conn = nn.Linear(self.hidden_size, num_classes, bias=False)
        self.output_conn.weight.data = output_errors
        self.output_conn.weight.requires_grad = False
        # self.output_conn.bias.data = torch.zeros_like(self.output_conn.bias.data)
        # self.act_out = Gaussian(mean=torch.Tensor(1), std=stdev)
        # self.output_act = nn.LogSoftmax(dim=1)
        self.output_act = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = len(x)
        # out = self.input_conn(x)
        out = self.act_hidden(x, batch_size)
        if not self.node:
            out = self.average_synapses(out)
            out = self.act_out(out, batch_size)
        out = self.output_conn(torch.transpose(out, 0, 1))
        out = self.output_act(out)
        return out

def check_memory(where=''):
    print(where)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print("     total      reserved       allocated       free")
    print(["{0:.4E}".format(thing) for thing in [t, r, a, f]])

def make_network(new_centres, new_mask, errors, old_model=False):
    models = {}
    new_params = []
    if old_model:
        for width, m, new_c, new_m, e in zip(parameter_settings, old_model, new_centres, new_mask, errors):
            # check_memory("mn")
            if not len(new_c):
                models['{}'.format(width)] = old_model['{}'.format(width)]
                new_params.append({'params': old_model['{}'.format(width)].parameters()})
                continue
            old_centres = old_model[m].act_hidden.mean
            old_weights = old_model[m].output_conn.weight.data
            old_mask = old_model[m].synapse_mask
            # check_memory("mnw")
            models['{}'.format(width)] = NeuralNet(num_classes,
                                                   centres=torch.vstack([old_centres, new_c]),
                                                   synapse_mask=torch.vstack([old_mask, new_m]),
                                                   output_errors=torch.hstack([old_weights, e]),
                                                   node=node,
                                                   tri=tri,
                                                   stdev=width).to(device)
            # check_memory("mnn")
            new_params.append({'params': models['{}'.format(width)].parameters()})
            # check_memory("mnp")

    else:
        for width in parameter_settings:
            models['{}'.format(width)] = NeuralNet(num_classes,
                                                   centres=new_centres,
                                                   synapse_mask=new_mask,
                                                   output_errors=errors,
                                                   node=node,
                                                   tri=tri,
                                                   stdev=width).to(device)
            new_params.append({'params': models['{}'.format(width)].parameters()})

    # lossFunction = nn.NLLLoss(reduction='none')
    lossFunction = nn.CrossEntropyLoss(reduction='none')
    optimize_all = optim.SGD(new_params,
                             lr=lr, momentum=momentum)
    return models, lossFunction, optimize_all

def neurogen_process(inputs, outputs, labels, loss):
    # centres = torch.stack([inputs for i in range(len(loss))])
    centres = inputs
    one_hot = torch.zeros([len(labels), num_classes])
    for i, l in enumerate(labels):
        one_hot[i, l] = 1
    errors = one_hot - torch.stack(outputs)
    triggered = (torch.max(torch.abs(errors), dim=2)[0] > error_threshold).nonzero(as_tuple=False)

    new_centres = [[] for i in range(len(loss))]
    new_errors = [[] for i in range(len(loss))]
    for [i, j] in triggered:
        new_centres[i].append(centres[j])
        new_errors[i].append(errors[i, j, :])
    mask = []
    for i in range(len(new_centres)):
        if len(new_centres[i]):
            new_centres[i] = torch.stack(new_centres[i])
            mask.append(torch.ones_like(new_centres[i]))
            new_errors[i] = torch.transpose(torch.stack(new_errors[i]), 0, 1)
        else:
            mask.append([])
    new_centres = new_centres
    new_errors = new_errors
    return new_centres, mask, new_errors


for images, labels in train_loader:
    images = images.reshape(-1, 784).to(torch.device(device))

    errors = torch.zeros([num_classes, batch_size_train])
    for i, l in enumerate(labels):
        errors[l, i] = 1

    models, lossFunction, optimize_all = make_network(new_centres=images,
                                                      new_mask=torch.ones_like(images),
                                                      errors=errors)
    break

training_losses = []
testing_accuracies = []

stop_growing = False

for epoch in range(num_epochs):
    check_memory("start")

    print(test_label)
    processed_masks = []
    for p in models:
        models[p].train()
    loss_ = [0 for i in range(len(parameter_settings))]
    # with torch.no_grad():
    for batch, (images, labels) in enumerate(train_loader):
        # check_memory("batch")
        print("Starting batch", batch+1, "/", len(train_loader))
        if not batch and not epoch:
            continue
        images = images.reshape(-1, 784).to(torch.device(device))

        output = []
        for p in models:
            output.append(models[p](images))
        # check_memory("forward")

        loss = []
        example_loss = []
        for out in output:
            loss_each = lossFunction(out, labels)
            example_loss.append(loss_each)
            loss.append(torch.mean(loss_each))
        # check_memory("loss")

        optimize_all.zero_grad()

        if learning and epoch:
            for l in loss:
                l.backward()
            optimize_all.step()
        # check_memory("learning")

        if not stop_growing and not epoch:
            centres, mask, errors = neurogen_process(images, output, labels, example_loss)
            # check_memory("process")

            models, lossFunction, optimize_all = make_network(new_centres=centres,
                                                              new_mask=mask,
                                                              errors=errors,
                                                              old_model=models)
            # check_memory("make")
        if (batch-1) % retest_rate == 0:
            with torch.no_grad():
                for p in models:
                    models[p].eval()
                correct = [0 for i in range(len(parameter_settings))]
                total = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, 784).to(torch.device(device))
                    out = []
                    for p in models:
                        out.append(models[p](images))
                    predicted = []
                    for o in out:
                        _, pred = torch.max(o, 1)
                        predicted.append(pred)
                    for i in range(len(parameter_settings)):
                        correct[i] += (predicted[i] == labels).sum().item()
                    total += labels.size(0)
                print(test_label)
                testing_accuracies.append(100 * np.array(correct) / total)
                for i in range(len(parameter_settings)):
                    print('Testing accuracy: {} %  {} n{}'.format(
                        np.array(testing_accuracies).astype(float)[:, i],
                        parameter_settings[i],
                        models['{}'.format(parameter_settings[i])].hidden_size))
        # check_memory("loss2")
        for i in range(len(parameter_settings)):
            loss_[i] += loss[i].detach()
        torch.cuda.empty_cache()

    for i in range(len(parameter_settings)):
        print("Epoch{}, Training loss:{} types:{}".format(epoch,
                                                             loss_[i] / len(train_loader),
                                                             parameter_settings[i]))
    training_losses.append(loss_)

    # Testing
    with torch.no_grad():
        for p in models:
            models[p].eval()
        correct = [0 for i in range(len(parameter_settings))]
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 784).to(torch.device(device))
            out = []
            for p in models:
                out.append(models[p](images))
            predicted = []
            for o in out:
                _, pred = torch.max(o, 1)
                predicted.append(pred)
            for i in range(len(parameter_settings)):
                correct[i] += (predicted[i] == labels).sum().item()
            total += labels.size(0)
        print(test_label)
        for i in range(len(parameter_settings)):
            print('Testing accuracy: {} %  {}'.format(100 * correct[i] / total, parameter_settings[i]))
        testing_accuracies.append(100 * np.array(correct) / total)

    if len(testing_accuracies) % 10 == 0:
        print("plotting")
        plt.figure()
        for i, p, in enumerate(models):
            print("\n", np.max(np.array(testing_accuracies).astype(float)[:, i]), p, models[p].hidden_size, "\n", np.array(testing_accuracies).astype(float)[:, i])
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

        print("done plotting")

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