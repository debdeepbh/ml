# Pytorch basics

[book](https://d2l.ai/chapter_preliminaries/ndarray.html)

# Install

```
pip3 install torch
```

# Topics

## Read/write with pandas [`rw.py`](rw.py)

- replacing missing values with the mean of the column `.fillna(value=value, inplace=True)`
- converting categorical variables (e.g. string) to indicator variables (one-hot) `pd.get_dummies(data, dummy_na=True)`
- extracting pandas table data segments as tensor etc `data.iloc[:, list].values`

## Tensor handling [`manipulation.py`](manipulation.py)

- reshaping `reshape`
- concatenating along axis `.cat((x,y), dim=1)`
- zero tensor of size matching another tensor `torch.zeros_like(another_tensor)`
- in-place addition `q[:] = x + y` as opposed to `q = x+y`

## Linear algebra [`linalg.py`](linalg.py)

- matrix-matrix product with `torch.mm(A, B)`
- matrix-vector product with `torch.mv(A, v)`
- matrix product of tensors with `torch.matmul(A, v)` or `torch.matmul(A,B)` (more general)
- operations over dimension `dim=` and keeping dimension `keepdims=True` 

## Auto differentiation [`autodiff.py`](autodiff.py)

- define a tensor as a variable of differentiation (or gradient) with `x.requires_grad_(True)` (in addition to as a point of evaluation)
- populate $x$ with the gradient of a functional $y  = f(x)$ w.r.t. $x$ evaluated at $x=x$ with `y.backward()` and retrieve it with `x.grad()`
- clean up the gradient value before computing a new gradient with `x.grad.zero_()`
- treat a function `y` of a `requires_grad_()` variable as constant with `y.detach()`

## Linear regression from scratch [`linreg.py`](linreg.py) 

- **Goal:** estimate $m, b$ to fit data $y = mX + b$, where $X$ is feature and $y$ is label
- create synthetic data using $y = mX + b + n$ where $n$ is normal(0,1) `torch.normal`
- a batch iterator shuffles indices of the data and returns (`yield`s) batches of data so that every call of the iterator returns a non-overlapping subset. Using `yield` instead of `return` makes a function behave like a `for` loop by remembering all the previous calls.
- **Prediction model:** `y_hat = m_est * X + b_est ` and **loss function** to minimize is 
$$
loss =  \sum_{minibatch} | y_{hat} - y|^2
$$

- **Minimization using stochastic gradient descent:** 

Initial guess:
$$
m = normal(0, 0.01)
\\
b = zeros
$$
Iteration: for each `epoch` (iteration of minimization step) and for each minibatch, 

1. compute the $loss on current minibatch$
1. compute $\nabla_{(m,b)} (loss on current minibatch)$ using backward differentiation
1.
$$
(m,b)_{new} = (m,b)_{old} - \Delta t \nabla_{(m,b)} (loss on current minibatch) / batchsize
\\
(m,b)_{new}.grad.zero_()
$$
where $\Delta t$ is the learning rate, gradient is computed using auto differentiation with $m, b$ as `requires_grad_(True)` with initialization of normal(0,0.01) and `zeros`, respectively.

- The stochastisticity comes from the randomness associated with picking the subsets (minibatches).
- The loss at each epoch is computed to be the loss for the last minibatch. Probabilistically, this would be the smallest loss among all other minibatches.

- At each epoch, the estimate for $(m,b)$ is improved (learned) $N/batchsize$ number of times, based on the estimate from the previous minibatch.


## Linear regression ready-made [`linalg_ready.py`](linalg_ready.py)


### Iterator

Use `next(iter(data))` to iterate over a chunk of data. For example,

```python
x = list(range(50))
it = iter(x)	# define an iterator
print(next(it))	# output: 0
print(next(it)) # output: 1
print(next(it)) # output: 2
```

The torch utility `data.DataLoader()` creates an iterator that allows creation of minibatches and also allows shuffles. Moreover, parallel loading is supported. See [`fashion.py`](fashion.py) for an example.

```python
from torch.utils import data
dd = data.DataLoader(x, batch_size=5, shuffle=False, num_workers=4)
it = iter(dd)
print(next(dd))	# output: [0, 1, 2, 3, 4]
print(next(dd))	# output: [5, 6, 7, 8, 9]
print(next(dd))	# output: [10, 11, 12, 13, 14]
```




### Neural network structure

- A single layer Sequential neural network with x-dim = 2 (feature) and y-dim =1 (label) can be created using

```python
net = nn.Sequential(nn.Linear(2,1))
```


- Initialize the weights and the bias of the first layer

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

Here, `net[0].weight` is `m` and `net[0].bias` is `b`.

- Define the loss function as mean-squared error

```python
loss = nn.MSELoss()
```

- Train using

```python
for epoch in 3:
    for X, y in data_iter:
        # l obtained through nn.MSELoss() already has the sum (then avg) of the loss over the minibatch
        l = loss(net(X) ,y)
        # same as setting zero_grad() to each parameter in net.parameters()
        trainer.zero_grad()
	# backward differentiation and store in net.parameters()
        l.backward()
        # step() will update the value of the loss-minimizer parameters
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

### Reading the loss-minimizing parameters

- Read the trained wights $w$ and bias $b$ using

```
w = net[0].weight.data
b = net[0].bias.data
```


## Downloading and viewing the fashion datasets [`fashion.py`](fashion.py)

- Download the dataset from `torchvision.datasets.FashionMNIST()`. Use `train=False` to get the testing data.
- Use `transform=tansforms.ToTensor()` to convert image data into a normalized (between 0 and 1) 32-bit floating point tensor.
- show the image using `imshow`

## Softmax regression [`fashion_softmax.py`](fashion_softmax.py)

- Goal: Given data $x \in \R^d$ with *one-hot* label $y \in \R^p$ (i.e, each $y = e_i$ for some $i$, where $e_i$ is a standard Euclidean basis), need to estimate weights $W \in \R^{d \times p}$ and bias $b \in \R^p$ such that
$$
y^{hat} = softmax { ( \sum_j x_j W_{jk} + b_k)_{k=1}^p }
$$
estimates $y$, where, for any $z \in \R^p$
$$
softmax(z) = z / \sum_{j=1}^p z_j .
$$

- For a minibatch of $n$ datapoints, $X \in \R^{n \times d}$ is the feature matrix (where each row contains a feature) and $Y \in \R^{n \times p}$ is the label matrix (each row contains an label). 

The likelihood function to *maximize* is therefore  (the explanation is questionable)
$$
L = \prod_{i=1}^n P(y^i | x^i)
$$
and the loss function to *minimize* is
$$
- \log (L) 
= - \sum_{i=1}^n \sum_{j=1}^p y_j^i \log (y^{hat,i}_j) 
$$
because (questionable explanation)
$$
P(y^i | x^i) = \prod_{j=1}^p e^{y_j \log (y^{hat}_j} .
$$

See another tutorial at [stanford tutorial](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)

- The derivative of the loss function $-\log L$ is given by
$$
\nabla_{w} -\log L=  - \sum_{i=1}^n x^i (y^i  - y^{hat,i}).
$$

### Code
- For 28x28 images with 10 possible labels, $W \in \R^{28*28 \times 10}$ and $b \in \R^{10}$. As before, they are initialized with normal(0, 0.01) and zeros, respectively.
- The cross entropy of $i$-th prediction $- y^i \log(y^{hat,i})$ is summed over the minibatch and is minimized using the stochastic gradient descent method. Each epoch runs $N/batchsize$ iterations of learning.
- The training accuracy (#correct prediction in the minibatch / batchsize) and test accuracy (#correct prediction on all test data/ size of test data) is printed after each epoch.


## Softmax with inbuilt functions [`fashion_softmax_ready.py`](fashion_softmax_ready.py)

- Sequentially define a neural network where the first layer flattens the `1x28x28`, then passes through a linear neural network with 10 outputs

```python
net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
```

- Initializing all the weights (including biases) in the entire sequential network (recursively) to a normal(0, 0.01)

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights) # recursively initializes all weights
```

Alternatively, we can initialize only the weights and biases of the second layer (`nn.Linear(28*28, 10)`) with normal(0,0.01) and zeros, respectively, with

```python
net[1].weight.data.normal_(0, 0.01)
net[1].bias.data.fill_(0)
```

- Define the loss function and training algorithm

```python
loss = nn.CrossEntropyLoss(reduction='mean')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

Note: `reduction='mean'` produces consistently better accuracy on test data than `sum`.

- Train using

```python
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    # computing loss on test data (appropriate when reduction='sum' is sum)
    for X, y in test_iter:
        l_acc += loss(net(X), y)
    print('loss on test data', l_acc)
```

- Compute accuracy (different metric than loss) by computing number of correct prediction to the total number of data points on the test data. To compute predicted label from softmax value (which is a vector of floats), use `argmax`.

- Compute total number of correct predictions on a minibatch of test data 

```python
def accuracy(y_hat, y):
    """Compute the number of correct predictions by converting softmax value to integer label"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat == y )
    return float(cmp.sum())
```

