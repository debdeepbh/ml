import torch
import math
from torch import nn

# solving ODE u'' + u = 0, u(0) = 0, u(\pi/2) = 3
# solution: u(x) = 3 \sin x
# Create Tensors to hold input and outputs.
X = torch.linspace(0, math.pi/2, 2000)
dx = X[1]- X[0]
X = X.reshape(-1,1)
print('X[0], X[-1]', X[0], X[-1])


# y = torch.sin(x)

net = nn.Sequential( 
                    nn.Linear(1,5),
                    # nn.ReLU(),
                    nn.Tanh(),
                    nn.Linear(5,1)
                    )

# n = 4
# # (size of data, number of features)
# X = torch.rand(size=(n,1))
# print('X', X)
# print('net(X)', net(X))

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = torch.tensor(1e-6, requires_grad=False)

# Define the ODE net
class ODENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( 
                            nn.Linear(1,5),
                            nn.Tanh(),
                            nn.Linear(5,1)
                            )
    
    def forward(self, x):
        y_hat = self.net(x)
        u_p = (y_hat[1:] - y_hat[:-1])/dx
        # u_pp = torch.diff(u_p, n=1, dim=-1, append=None)/dx
        # return y_hat
        return u_p

class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

class PlainDiff(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input[1:] - input[:-1]

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

onet = ODENet()
print('Net comprises', sum([p.numel() for p in onet.parameters()]), 'parameters')


model = onet
# model = net

for t in range(10):
    y_pred = model(X)

    # P3 = LegendrePolynomial3.apply
    # P3(y_pred)

    PD = PlainDiff.apply

    # forward difference of a vector, prepend zero (in the beginning) so that 
    # u_p = torch.diff(y_pred, n=1, dim=-1, prepend=torch.tensor([0]), append=None)/dx
    # u_pp = torch.diff(u_p, n=1, dim=-1, prepend=torch.tensor([0]), append=None)/dx
    u_p = torch.diff(y_pred, n=1, dim=-1, append=None)/dx
    u_pp = torch.diff(u_p, n=1, dim=-1, append=None)/dx
    ode = u_pp + u_p

    # u_p = torch.autograd.grad(y_pred, X, retain_graph=True, create_graph=True)[0]
    # ode = u_p

    # ode = y_pred**2

    loss_int = loss_fn(ode, torch.zeros_like(ode))
    loss_bdry = loss_fn( torch.tensor([X[0], X[-1]]), torch.tensor([0, 3]))
    loss = loss_int + loss_bdry
    # if t % 100 == 99:
    print(t, loss.item())

    # loss.backward()
    for param in model.parameters():
        print('param.grad', param.grad)

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    # for param in model.parameters():
    #     print('param.grad', param.grad)

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
