import  torch
# source: https://d2l.ai/chapter_preliminaries/ndarray.html


x = torch.arange(12, dtype=torch.float32).reshape((3,4))
print(x)
print(x.shape)
print(x.numel())

y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

z = torch.cat((x, y), dim=0)
w = torch.cat((x, y), dim=1)

print('z',z)

print('w',w)

p = torch.zeros_like(x)
q = torch.zeros_like(x)
# id() gives the memory location
id_of_p = id(p)
id_of_q = id(q)

p = x + y

new_id_of_p = id(p)
print('for p', id_of_p == new_id_of_p)


# in-place addition to save space
# compared to q = x+y, which does not save space
q[:] = x + y
new_id_of_q = id(q)
print('for q', id_of_q == new_id_of_q)

