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
    # caution: watch out for zero-division, which most likely won't happen
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
    # comparison after converting to the same datatype
    cmp = (y_hat.type(y.dtype) == y)
    # sum to get the number of correct predictions
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode

    # for storing both the number of correct predictions and the number of predictions, respectively. 
    # Both will be accumulated over time as we iterate over the dataset.
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    # number of correct predictions / total number of predictions
    return metric[0] / metric[1]

lr = 0.1

def sgd(params, lr, batch_size): 
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

def net(X):
    """ the soft regression model, similar to a neural net
    """
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def train_epoch(net, train_iter, loss, updater):
    """ train a single epoch
    """
    # if the object net is a pytorch neural network module, then use in-built function train()
    if isinstance(net, torch.nn.Module):
        print('yes, net is a torch module')
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)

    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        # here, loss function will be cross_entropy()
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy: accuracy = #correct predictions/#total predictions
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print('training loss, training accuracy', train_metrics, 'test_accuracy', test_acc)

    train_loss, train_acc = train_metrics

    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

num_epochs = 10
train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
