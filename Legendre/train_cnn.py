import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization


def check_memory(where=''):
    print(where)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print("   total     reserved     allocated     free")
    print(["{0:.2E}".format(thing) for thing in [t, r, a, f]])


if __name__ == '__main__':

    input_size = 28 * 28
    num_classes = 10
    num_epochs = 100
    lr = 0.05
    momentum = 0.9

    hidden_size = [200]
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
        # ['relu', 'tanh', 'sig'],
        # ['relu', 'gelu', 'lrelu'],
        # ['relu', 'tanh', 'sig', 'gelu', 'lrelu'],
        # ['relu', 'tanh', 'sig', 'smin', 'smax', 'gelu', 'lrelu']
    ]
    batch_size = 128
    trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, generator=torch.Generator(device=device))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, generator=torch.Generator(device=device))

    cnn = CNN()

    # lossFunction = nn.NLLLoss()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)

    training_losses = []
    testing_accuracies = []

    cnn.train()

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # gives batch data, normalize x when iterate train_loader
            # b_x = Variable(images)  # batch x
            # b_y = Variable(labels)  # batch y
            output = cnn(images.to(torch.device(device)) - 0.5)[0]
            loss = loss_func(output, labels)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()  # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        for i in range(len(neuron_types)):
            print("Epoch{}, Training loss:{} types:{}".format(epoch,
                                                              loss / len(train_loader),
                                                              neuron_types[i]))
        training_losses.append(loss)

        # Testing
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(torch.device(device)) - 0.5
                b_x = Variable(images)  # batch x
                b_y = Variable(labels)  # batch y
                out = cnn(b_x)[0]
                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            print('Testing accuracy: {} %'.format(100 * correct / total))
            testing_accuracies.append(100 * np.array(correct) / total)

    print("training:")
    print(np.array(training_losses).astype(float)[:, i])
    # print(training_losses)
    print("testing")
    print("\n", np.array(testing_accuracies).astype(float)[:, i])

    test_label = "test_acc{}".format(testing_accuracies[-1])
    torch.save(cnn, 'mnist0.5 CNN sigmoid {}.pt'.format(test_label))

    print('done')
