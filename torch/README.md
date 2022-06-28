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

Data iterator library comes from

```python
from torch.utils import data
```

It provides an iterator that loads inside an iterator function

```python
def load_array(data_arrays, batch_size):  
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=True)
```

and called using

```python
data_iter = load_array((features, labels), batch_size)
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

- Download the dataset from `torchvision.datasets.FashionMNIST()` 
- show the image using `imshow`
