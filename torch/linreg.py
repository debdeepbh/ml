import numpy as np
import torch
import matplotlib.pyplot as plt
import random

# source: https://d2l.ai/chapter_linear-networks/linear-regression-scratch.html


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

# print(features)
# plt.scatter(features[:, 0], features[:,1], c=labels)
# plt.show()

#######################################################################
# create batches

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    # every batch_size element in the shuffled dataset
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        # yield (instead of return) maintains the last state of the function; see function call
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

## Print a single batch worth of data
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# initial guess for w is random; initial guess for b is zero
# w and b will be treated as variables of differentiation: requires_grad
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# predicted y: y_hat
def linreg(X, w, b): 
    """The linear regression model: y = Xw + b"""
    return torch.matmul(X, w) + b

# loss function
def squared_loss(y_hat, y):
    """Squared loss: 1/2 * (yhat - y)"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# """Minibatch stochastic gradient descent."""
def sgd(params, lr, batch_size): 
    """Minibatch stochastic gradient descent."""

    # by default, any variable depending on variables of differentiation will have use_grad=True
    # turn off use_grad for any variable defined in the block
    with torch.no_grad():
        # params = [w , b]
        for param in params:
            # Because our loss is calculated as a sum over the minibatch of
            # examples, we normalize our step size by the batch size,
            # so that the magnitude of a typical step size does
            # not depend heavily on our choice of the batch size.
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#######################################################################
# training

lr = 0.03   # learning rate
num_epochs = 3
net = linreg    # name of the model function
loss = squared_loss # name of the loss function 

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # a vector of size batch_size containing point wise error
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        # at this point, w.grad and b.grad will be available for use in sgd()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient

    # turn off use_grad for any variable defined in the block
    # by default, any variable depending on variables of differentiation will have use_grad=True
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


