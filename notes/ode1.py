import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np 

""" Solving ODE on [-2,2]
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
        self.net = nn.Sequential( 
                            nn.Linear(1,5),
                            nn.Tanh(),
                            nn.Linear(5,1)
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

    # y' = - 2x*y
    eq = dydx + 2.* x * y 
    # y(x=0) = 1
    ic = model(torch.tensor([0.])) - 1.   
    return torch.mean(eq**2) + ic**2

loss_func = ODE

# Define the optimization
# opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.99,nesterov=True) # Equivalent to blog
opt = optim.Adam(model.parameters(),lr=0.1,amsgrad=True) # Got faster convergence with Adam using amsgrad

# Define reference grid 
x_data = torch.linspace(-2.0,2.0,401,requires_grad=True)
x_data = x_data.view(401,1) # reshaping the tensor

# Iterative learning
epochs = 1000
for epoch in range(epochs):
    opt.zero_grad()
    y_pred = model(x_data)
    dy_pred_dx = ODE(x_data, y_pred)
    loss = dy_pred_dx
    # loss = torch.abs(dy_pred_dx)

    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

# Plot Results
plt.plot(x_data.data.numpy(), np.exp(-x_data.data.numpy()**2), label='exact')
y_data = model(x_data)
plt.plot(x_data.data.numpy(), y_data.data.numpy(), label='approx')
plt.legend()
plt.show()
