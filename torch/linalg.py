import torch

# source: https://d2l.ai/chapter_preliminaries/linear-algebra.html

# matrix
A = torch.arange(20).reshape(5, 4)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)

# transpose
print(B == B.T)

# element-wise multiplication
C = B * B
print(C)

# matrix-matrix product
BB = torch.mm(B,B)
print(BB)

# 3-tensor
X = torch.arange(24).reshape(2, 3, 4)
print('X',X)

# column-sum, reduce to 1-tensor
Bsum = B.sum(axis=1)
print('Xsum',Bsum)

# column-sum, but keep it as 2-tensor
Bsum_same = B.sum(axis=1, keepdims=True)
print('Bsum_same',Bsum_same)

# matrix-vector product
Bv = torch.mv(B, Bsum)
print(Bv)

# norm
norm = torch.abs(Bv).sum()
print(norm)
