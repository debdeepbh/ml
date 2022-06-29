import torch
from torch import nn
import torchvision

import matplotlib.pyplot as plt

from torch.utils import data
from torchvision import transforms

# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()

def load_data_fashion_mnist(batch_size=256, resize=None, num_workers=4):
    """Download the Fashion-MNIST dataset and then load it into memory."""

    # converts into a tensor of size 28x28
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans, download=True)

    # return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
    #         data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))


train_iter, test_iter = load_data_fashion_mnist()
# data_iter = load_data_fashion_mnist()
# print(data_iter)

# a fully-connected nn: first flatten the 28x28 image into a vector, then pass it to
# a linear (affine?) network with 10 outputs
net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))

# net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10), nn.Linear(10, 3)) # add as many Sequentially

# This is apparently the standard way to `apply` initialization to a neural network recursively
def init_weights(m):
    # if type(m) == nn.Linear:  # this works too
    if isinstance(m, nn.Linear):
        # this initializes both weight and bias to normal(0, 0.01)
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights) # recursively initializes all weights

# ## Alternatively, this works
# net[1].weight.data.normal_(0, 0.01)
# net[1].bias.data.fill_(0)

print('weight', net[1].weight)
print('bias', net[1].bias)
print('params', net.parameters)

# nn.CrossEntropyLoss() with `reduction='none'` is vector-valued, autodiff cannot be computed
# Therefore, use nn.CrossEntropyLoss(reduction='sum' or 'mean') or while training do loss.sum().backward() or loss.mean().backward()
# loss = nn.CrossEntropyLoss(reduction='none')
# reduction = 'mean'
reduction = 'sum'
loss = nn.CrossEntropyLoss(reduction=reduction)
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

def accuracy(y_hat, y):
    """Compute the number of correct predictions by converting softmax value to integer label"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat == y )
    return float(cmp.sum())

class TestMetric(object):
    def __init__(self, reduction):
        self.reduction = reduction

    def reset(self):
        self.datasize = 0
        self.testloss_sum = 0.0
        self.total_correct = 0.0

    def testloss(self):
        return self.testloss_sum /self.datasize 
    def testaccuracy(self):
        return self.total_correct /self.datasize 

    def collect(self, net, X, y):

        # compute loss
        if self.reduction == 'sum':
            self.testloss_sum += loss(net(X), y)
        elif self.reduction == 'mean':
            self.testloss_sum += loss(net(X), y) * len(y)

        # compute accuracy
        self.total_correct += accuracy(net(X), y)

        self.datasize += len(y)
        

num_epochs = 20
TM = TestMetric(reduction)
for epoch in range(num_epochs):
    print('epoch', epoch)
    # i=0
    TM.reset()
    for X, y in train_iter:

        # print('iteration', i)
        # i += 1

        l = loss(net(X) ,y)
        trainer.zero_grad()
        # nn.CrossEntropyLoss() with `reduction='none'` is vector-valued. Need to reduce to a scalar either via sum or mean
        # l.mean().backward()
        l.backward()
        trainer.step()

    ## total loss on test data
    # l_acc = 0
    # for X, y in test_iter:
    #     l_acc += loss(net(X), y)
    # print('loss on test data', l_acc)

    for X, y in test_iter:
        TM.collect(net, X, y)
    print('loss on test data', TM.testloss())
    print('accuracy on test data', TM.testaccuracy())

