import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np 
import math
import copy

""" 
Attempt to define an autograd function with custom derivate
ODE will be in the model, not the loss function this time
Solving ODE on [-2,2]
    y' = - 2x*y
    y(x=0) = 1


    https://stackoverflow.com/questions/58839721/how-to-define-a-loss-function-in-pytorch-with-dependency-to-partial-derivatives
"""

# Define the NN model to solve the problem
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(1,10)
        self.lin2 = nn.Linear(10,1)

    def forward(self, x):
        y = torch.sigmoid(self.lin1(x))
        z = torch.sigmoid(self.lin2(y))
        return z

# Define the ODE net
class Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        n = 5
        self.net = nn.Sequential( 
                            nn.Linear(1,n),
                            nn.Tanh(),
                            nn.Linear(n,1),
                            )
    
    def forward(self, x):
        y_hat = self.net(x)
        # u_p = (y_hat[1:] - y_hat[:-1])/dx
        # u_pp = torch.diff(u_p, n=1, dim=-1, append=None)/dx
        return y_hat

# model = Model()
model = Model_2()


# Define loss_function from the Ordinary differential equation to solve
def ODE(x,y):
    dydx, = torch.autograd.grad(y, x, 
    grad_outputs=torch.ones_like(x),
    create_graph=True, 
    retain_graph=True,
    # allow_unused=True,    # suggested by stackoverflow, but slows down computation
                                )

    # # y' = - 2x*y # y(x=0) = 1
    # eq = dydx + 2.* x * y 
    # ic = model(torch.tensor([0.])) - 1.   

    dydx2, = torch.autograd.grad(dydx, x, 
    grad_outputs=torch.ones_like(x),
    create_graph=True, 
    retain_graph=True,
    # allow_unused=True,    # suggested by stackoverflow, but slows down computation
                                )
    # # solving ODE u'' + u = 0, u(0) = 0, u(\pi/2) = 3
    eq = dydx2 + dydx
    ic1 = model(torch.tensor([0.])) - 0.   
    ic2 = model(torch.tensor([math.pi/2])) - 3.   

    # return torch.mean(eq**2) + ic**2
    return torch.mean(eq**2) + ic1**2 + ic2**2

# loss_func = ODE
loss_fn = torch.nn.MSELoss(reduction='sum')

# Define the optimization
# opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.99,nesterov=True) # Equivalent to blog
opt = optim.Adam(model.parameters(),lr=0.1,amsgrad=True) # Got faster convergence with Adam using amsgrad

# Define reference grid 
# N = 401
N = 10
# N = 30
# x_data = torch.linspace(-2.0,2.0,N,requires_grad=True)
# x_data = torch.linspace(0,math.pi/2,N,requires_grad=True)
x_data = torch.linspace(0,math.pi/2,N,requires_grad=True)
x_data = x_data.reshape(N,1) # reshaping the tensor

dx = (x_data[1] - x_data[0]).clone().detach().numpy()[0]

class MyDiff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        # finite difference
        dx = x_data[1] - x_data[0]

        ctx.save_for_backward(input, dx)

        d1 = torch.zeros_like(input)
        fd1 = (input[1:] - input[:-1])/dx
        d1[:-1] = fd1

        d2 = torch.zeros_like(input)
        fd2 = (d2[1:] - d2[:-1])/dx
        d2[:-1] = fd2


        # d1, = torch.autograd.grad(input, x_data, 
        # grad_outputs=torch.ones_like(x_data),
        # create_graph=True, 
        # retain_graph=True,
        # # allow_unused=True,    # suggested by stackoverflow, but slows down computation
        #                             )
        #
        #
        # d2, = torch.autograd.grad(d1, x_data, 
        # grad_outputs=torch.ones_like(x_data),
        # create_graph=True, 
        # retain_graph=True,
        # # allow_unused=True,    # suggested by stackoverflow, but slows down computation
        #                             )

        # y = copy.deepcopy(input)
        # y.detach().numpy().reshape(-1)
        # d1 = np.gradient(y, dx)
        # d2 = np.gradient(d1, dx)

        eq = d2 + d1

        return eq

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # _, = ctx.saved_tensors
        # Derivative times a vector grad_output
        out = torch.zeros_like(grad_output)
        # fd1 = (grad_output[1:] - grad_output[:-1])/dx
        fd1 = (grad_output[:-1] - grad_output[1:])/dx
        out[:-1] = fd1

        # out, = torch.autograd.grad(grad_output, x_data, 
        # grad_outputs=torch.ones_like(x_data),
        # create_graph=True, 
        # retain_graph=True,
        # # allow_unused=True,    # suggested by stackoverflow, but slows down computation
        #                             )


        # y = copy.deepcopy(grad_output)
        # y = y.detach().numpy().reshape(-1)
        # d1 = np.gradient(y, dx)

        return out

# y_pred = model(x_data)
# dydx, = torch.autograd.grad(y_pred, x_data, 
# grad_outputs=torch.ones_like(x_data),
# create_graph=True, 
# retain_graph=True,
# # allow_unused=True,    # suggested by stackoverflow, but slows down computation
#                             )
#
# print('x_data, y_pred', x_data.T, y_pred.T)
# print('dydx', dydx.T)
#
#
# input = y_pred
# dx = x_data[1] - x_data[0]
# # print('dx', dx)
# d1 = torch.zeros_like(input)
# diff = (input[1:] - input[:-1])
# # print('diff', diff)
# fd1 = diff/dx
# d1[:-1] = fd1
#
# print('y_pred.data.numpy().reshape(-1)', y_pred.data.numpy().reshape(-1))
# d1 = np.gradient(y_pred.data.numpy().reshape(-1), dx.data.numpy().reshape(-1)[0])
# print('d1', d1.T)
#
# plt.plot(x_data.data.numpy(),dydx.data.numpy(), label='dydx')
# # plt.plot(x_data.data.numpy(),d1.data.numpy(), label='d1')
# plt.plot(x_data.data.numpy(),d1, label='d1')
#
# plt.legend()
# plt.show()
# plt.close()
#
#
# import sys
# sys.exit(0)


# Iterative learning
# epochs = 1000
epochs = 5000
for epoch in range(epochs):
    opt.zero_grad()


    # y_pred = model(x_data)
    # dy_pred_dx = ODE(x_data, y_pred)
    # loss = dy_pred_dx

    y_pred = MyDiff.apply(model(x_data))
    loss = loss_fn(y_pred, torch.zeros_like(y_pred)) + loss_fn(torch.tensor([y_pred[0], y_pred[-1]]), torch.tensor([0, 3]))

    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

# Plot Results
# plt.plot(x_data.data.numpy(), np.exp(-x_data.data.numpy()**2), label='exact')
plt.plot(x_data.data.numpy(), 3 * np.sin(x_data.data.numpy()), label='exact')
y_data = model(x_data)
plt.plot(x_data.data.numpy(), y_data.data.numpy(), label='approx')
plt.legend()
plt.show()
