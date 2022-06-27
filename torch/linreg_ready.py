import numpy as np
import torch
from torch import nn
from torch.utils import data

# source: https://d2l.ai/chapter_linear-networks/linear-regression-concise.html

#######################################################################
# generate data

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# import matplotlib.pyplot as plt
# print(features)
# plt.scatter(features[:, 0], features[:,1], c=labels)
# plt.show()

#######################################################################
# create batches

def load_array(data_arrays, batch_size, shuffle=True):  
    """Construct a PyTorch data iterator."""
    # the asterisk is used to denote a variable number of arguments
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

## print the first batch
# print(next(iter(data_iter)))

# model: single layer Sequential neural network
# model: 2 = x-dim; 1 = y-dim
net = nn.Sequential(nn.Linear(2,1))

# initialize the weights and the bias of the first layer
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss function; default: reduction='mean'; 
loss = nn.MSELoss()

# training algorithm: SGD; parameters to optimize: net.parameters(); learning rate = lr
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# training process 
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # l obtained through nn.MSELoss() already has the sum (then avg) of the loss over the batch
        l = loss(net(X) ,y)
        # print('l=',l)
        # trainer.zero_grad() grad is same as setting zero_grad() to each parameter in net.parameters()
        trainer.zero_grad()
        l.backward()
        # step() will update the value of the minimizer parameters
        trainer.step()
    # again, nn.MSELoss() computes the mean of all the errors
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
