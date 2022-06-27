import torch

# source: https://d2l.ai/chapter_preliminaries/autograd.html
# autodiff: https://en.wikipedia.org/wiki/Automatic_differentiation
# autograd: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# length-4 vector
x = torch.arange(4.0)
print('x', x)  # The default value is None
# treat x as a variable for differentiation
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
print('x.grad', x.grad)  # The default value is None

# gradient computation of (2 x.x) via backpropagation
# y is now a function of (variable of diff) x
y = 2 * torch.dot(x, x)
print('y', y)
# gradient of y wrt variable x is computed and is stored in x (compare syntax of tensorflow: x_grad = t.gradient(y, x))
print('performing backward differentiation of y wrt x evaluated at x. Should be 4x.')
y.backward()

# compare tensorflow syntax: print(x_grad)
print('gradient of y evaluated at x:', x.grad)

# print('Clearing up x')
# x.grad.zero_()  # reset the gradient value
k = 3 * torch.dot(x, x)
print('k', k)
k.backward()
print('gradient of k evaluated at x without cleaning up x.grad:', x.grad)

# gradient computation of sum(x) wrt x
print('Clearing up x')
x.grad.zero_()  # reset the gradient value
y = x.sum()
y.backward()
print('gradient of y evaluated at x:', x.grad)

# detach
x.grad.zero_()
# y = y(x)
y = x * x
# treat u as a constant
u = y.detach()
print('u', u)
# z = z(x), where y is a constant
z = u * x

z.sum().backward()
print('x',x)
## This will output u  as opposed to 3x^2 (since u is considered constant)
print('gradient of z evaluated at x:', x.grad)
