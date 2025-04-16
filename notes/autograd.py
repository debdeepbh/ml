import torch
x = torch.linspace(-10, 10, 5, requires_grad=True)
ones = torch.ones_like(x)
z = x **2
# Compute the derivatives
grads = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=ones)

print('x', x)
print('ones', ones)
print('grads', grads)
