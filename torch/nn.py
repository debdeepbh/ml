import torch
from torch import nn
from torch.nn import functional as F

# net = nn.Sequential( nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
#
X  = torch.rand(2, 20)
#
# print(net(X))

# custom block

class MLP(nn.Module):
    """ Alternative to nn.Sequential: A custom nn with two fully connected layers"""
    def __init__(self):
        super().__init__()  # initializes the parent class `nn.Module` so that net(X) is well-defined (via the nn.Module feature)
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    def forward(self, X):
        return self.out( F.relu(self.hidden(X)))

net = MLP()

print(net(X))


# another custom block with parameters
class FixedHidenMLP(nn.Module):
    """docstring for FixedHidenMLP"""
    def __init__(self):
        super(FixedHidenMLP, self).__init__()
        self.rand_weight = torch.rand( (20,20), requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        """forward propagation
        :X: TODO
        :returns: some arbitrary value |c_2 . relu( c_1 . (w . X) + 1 )|_{l_1} , return max {1, out/2}
        """

        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)

        while X.abs().sum() > 1:
            X /= 2

        return X.sum()
        
# can call with
net = FixedHidenMLP()
print(net(X))

        
## can treat the newly created nn.Module elements  as regular nn modules
# mynet = nn.Sequential( nn.Linear(16, 20), FixedHidenMLP() )
# print(mynet(X))


## layer without weights. e.g. X - X.mean()

class CenteredLayer(nn.Module):
    """docstring for CenteredLayer"""
    def __init__(self):
        super(CenteredLayer, self).__init__()
        
    def forward(self, X):
        return X - X.mean()

net = nn.Sequential( nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4,8))
print(Y.mean())


## Custom layer with parameters
class MyLinear(nn.Module):
    """Custom linear layer"""
    def __init__(self, in_units, out_units):
        super(MyLinear, self).__init__()
        # nn.Parameter treats the variable with with_grad=True
        self.weight = nn.Parameter(torch.randn(in_units, out_units))
        self.bias = nn.Parameter(torch.randn(out_units,))

    def forward(self, X):
        """TODO: Docstring for forward.

        :X: TODO
        :returns: TODO

        """
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print('Weights of MyLinear custom linear layer', linear.weight)
print('Biases of MyLinear custom linear layer', linear.bias)


        
