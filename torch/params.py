import torch
from torch import nn

net = nn.Sequential( nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))
X = torch.rand(size=(2,4))

print(net(X))

## Use net.state_dict() to show the parameters
print('Show parameters of the third neural network in the sequence:', net[2].state_dict())
print('print the bias data of 3rd net:', net[2].bias.data)

# initialize the usual way
def init_constant(m):
    if type(m) == nn.Linear:
	# set constant 1 to each parameter
	# compare with nn.init.normal_(m.weight, mean, std)
        nn.init.constant_(m.weight, 1)
        # set constant zero
        nn.init.zeros_(m.bias)
# usual net.apply()
net.apply(init_constant)

print('weights now', net[0].weight.data[0], net[0].bias.data[0])


### Shared layers: same weights used twice
# We need to give the shared layer a name so that we can refer to its parameters
shared = nn.Linear(8, 8)
# Same weights used for 2 and 4th module
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print('net[2] and net[4] weights are the same?', print(net[2].weight.data[0] == net[4].weight.data[0])

# edit the parameter value of one shared layer
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the # same value
print('net[2] and net[4] weights are the same?', print(net[2].weight.data[0] == net[4].weight.data[0])


