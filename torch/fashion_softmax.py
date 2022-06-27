import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils import data
from torchvision import transforms

# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()

def load_data_fashion_mnist(batch_size=256, resize=None, num_workers=4):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))

# convert label names to label indices
def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#######################################################################

train_iter, test_iter = load_data_fashion_mnist()

# num_inputs (features) = 28x28 = 784 (will flatten the image); num_outputs = 10 labels (will convert to one-hot notation)
num_inputs = 28 * 28
num_outputs = 10

# initialize the weights, set them as variables of differentiation
# Model: yhat = softmax( W \tensor X + b) where W (28*28, 10), X (28*28,1), b(10,1)
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    """ softmax(X)_j = exp(X_j)/ sum_j exp(X_j) """
    X_exp = torch.exp(X)
    expsum = X_exp.sum(1, keepdim=True)
    # caution: watch out for zero-division
    return X_exp / expsum

def cross_entropy(y_hat, y):
    """cross-entropy loss function l(y, yhat) = - \sum_j  y_j * log(yhat_j) 
    :y_hat: array of predictions of size 10
    :y: array of scalar labels between 0 to 9
    :returns: 
    """
    # every scalar-valued in input array y is converted to a length-10 array in the one-hot notation
    return -torch.log(y_hat[range(len(y_hat)), y])

# a sample minibatch with 2 data points
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, -0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y]) # output: tensor([0.1, 0.5])

def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    :y_hat: array of vector-valued (one-hot) predictions
    :y: array of scalar-valued predictions
    """
    # if the minibatch has more than one data point
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # pick the largest probability as the predicted category
        y_hat = y_hat.argmax(axis=1)
    # comparison of the same datatypes
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
