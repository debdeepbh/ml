import torch
x = torch.linspace(-10, 10, 5, requires_grad=True)
x = x.reshape(-1,1).pow(torch.tensor([1, 2, 3]))
print('x', x)
A = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 1, 1.]])
z = torch.matmul(A, x)
print('z', z)
# Compute the derivatives
ones = torch.ones_like(z)
grads = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=ones)

print('ones', ones)
print('grads', grads)
