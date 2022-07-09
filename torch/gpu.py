import torch
from torch import nn

torch.device('cpu')
torch.device('cuda')
torch.device('cuda:1')

print('available number of gpus;', torch.cuda.device_count())
print('cuda available:', torch.cuda.is_available())

def try_gpu(i=0):
    """return gpu(i) if exists, otherwise return cpu().

    :i: TODO
    :returns: TODO

    """
    if torch.cuda.device_count() >= i+1:
        return torch.cuda.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """Trturn all avilable GPUs or [cpu(),] if no GPU exists.
    :returns: TODO

    """
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) ]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

x = torch.tensor([1,2,3])
print('tensor x resides in', x.device)

z = x.cuda(0)
print('tensor z resides in', z.device)
