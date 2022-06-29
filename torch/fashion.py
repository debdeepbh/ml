import torch
import torchvision
import matplotlib.pyplot as plt


from torch.utils import data
from torchvision import transforms

# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()

# download the data
mnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, transform=trans, download=True)

# description: Fashion-MNIST consists of images from 10 categories, each
# represented by 6000 images in the training dataset and by 1000 in the test dataset. 

print('train len', len(mnist_train), 'test len', len(mnist_test))
print('first train data label', mnist_train[0][1])

print('Shape of each image', mnist_train[0][0].shape) # [1, 28 , 28] single-channel 28x28

# convert label names to label indices
def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#######################################################################
# show some images
batch_size = 64
num_workers = 4 # parallel reading
X, y = next(iter(data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)))
y_labels = get_fashion_mnist_labels(y)

# add a subplot with no frame
rows = 8
for i in range(batch_size):
    plt.subplot(int(batch_size/rows), rows, i+1, frameon=False)
    plt.imshow(X[i].reshape([28,28]))
    plt.title(y_labels[i])
    plt.axis('off')
plt.show()

