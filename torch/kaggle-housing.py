
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt


train_data = pd.read_csv('data/kaggle-housing/train.csv', na_values = '-')
test_data  = pd.read_csv('data/kaggle-housing/test.csv', na_values  = '-')
## Replace NaN with 0, in place
# df.fillna(0, inplace=True)

## show all the columns
# print(train_data.columns)
# print(test_data.columns)
#
# show size
print('train data size', train_data.shape)
print('test data size', test_data.shape)

# print some data
print(train_data.iloc[0:5, [0,1, 2, 3, -3, -2, -1]])

# combine train and test data (except id and sales price)
alldata = pd.concat( (train_data.iloc[:, 1:-1], test_data.iloc[:,1:]))

print('number of features before', alldata.shape)

# pick the numeric features (all that are not objects); .index returns the names of the fields
# numeric_features = all_features.dtypes[all_features.dtypes != 'object']
numeric_features = alldata.dtypes[alldata.dtypes != 'object'].index
# print('numeric features', numeric_features)

# normalize to mean 0 and std 1
# pandas .mean() and .std() is done columnwise by default
alldata[numeric_features] = (alldata[numeric_features] - alldata[numeric_features].mean() ) / alldata[numeric_features].std()

# data[numeric_features].fillna(0, inplace=True)
alldata = alldata[numeric_features].fillna(0)

## check  that mean and std are applied
# print('mean and std column wise', numeric_data.mean(), numeric_data.std())
# print('mean and std column wise', all_features[numeric_features].mean(), all_features[numeric_features].std())

# one-hot encoding fields for non-numeric data; pd.get_dummies
# dummy_na creates an indicator for NA values
alldata = pd.get_dummies(alldata, dummy_na=True)

print('number of features after', alldata.shape)


n_train = train_data.shape[0]

# get numeric matrix from total data using train and test
train_features = torch.tensor( alldata[:n_train].values, dtype=torch.float32)
test_features = torch.tensor( alldata[n_train:].values, dtype=torch.float32)

# train_labels = torch.tensor( train_data.iloc[:,-1].values.reshape(-1,1), dtype=torch.float32)
train_labels = torch.tensor( train_data['SalePrice'].values.reshape(-1,1), dtype=torch.float32)
print('train labels', train_labels)

# number of input features
n_features = train_features.shape[1]

print('number of features', n_features)

# net
def get_nn():
    net = nn.Sequential( nn.Linear(n_features,1) )
    return net

init_std = 0.01

def init_weights(m):
    # if type(m) == nn.Linear:  # this works too
    if isinstance(m, nn.Linear):
        # this initializes both weight and bias to normal(0, 0.01)
        nn.init.normal_(m.weight, std=init_std)


# loss
loss = nn.MSELoss()

# another measure of error
def log_rmse(net, features, labels):
    """log-rmse = rmse of log of ratio of prediction and label: sqrt( sum_i ((log y_i - log y_hat_i)^2) / n)

    :net: TODO
    :features: TODO
    :labels: TODO
    :returns: TODO

    """

    # clamp all elements to the range [min, max] = [1, inf] (to stabilize log or something)
    clipped_prediction = torch.clamp( net(features), 1, float('inf') )

    # rmse of log of error
    rmse = torch.sqrt( loss(torch.log(clipped_prediction), torch.log(labels) ) )

    # return the value
    return rmse.item()


def attempt_log_rmse(predictions, labels):
    """ attempted custom loss function for training; to be differentiable, the operations have to be torch operations and must output a scalar.
    """
    clipped_prediction = torch.clamp(predictions, 1, float('inf') )
    rmse = torch.sqrt( torch.sum(torch.pow(torch.log(clipped_prediction) - torch.log(labels), 2), dim=0, keepdim=False) / labels.shape[0] )
    return rmse


# data iterator
def load_array(data_arrays, batch_size, shuffle=True):  
    """Construct a PyTorch data iterator."""

    # the asterisk is used to denote a variable number of arguments
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)

def train(net, train_features, train_labels, test_features, test_labels,
        num_epochs, learning_rate, weight_decay, batch_size):
    """

    :net: TODO
    :train_features: TODO
    :train_labels: TODO
    :test_features: TODO
    :test_labels: TODO
    :returns: TODO

    """

    # will contain errors
    train_ls, test_ls = [], []

    # iterator
    train_iter = load_array( (train_features, train_labels), batch_size)

    # using Adam instead of sdg; Adam is more stable wrt initial data
    optimizer = torch.optim.Adam( net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD( net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:

            # 1.set zero gradient
            optimizer.zero_grad()
            # 2. compute the loss
            l = loss(net(X), y)
            # l = attempt_log_rmse(net(X), y)
            # 3. backward differentiation
            l.backward()
            # 4. take iteration step with learning rate
            optimizer.step()

        # training loss for the epoch
        train_ls.append( log_rmse(net, train_features, train_labels) )

        # test loss for the epoch
        # prediction will not have any test data
        if test_labels is not None:
            test_ls.append( log_rmse(net, test_features, test_labels) )

    return train_ls, test_ls

def plot(train_ls, test_ls, show=True, close=True):
    """TODO: Docstring for plot.

    :train_ls: TODO
    :test_ls: TODO
    :returns: TODO

    """
    n = len(train_ls)
    plt.plot(range(n), train_ls, label='training loss')
    plt.plot(range(n), test_ls, label='test loss')
    plt.title(loss)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    #plt.xlim(float(xx[0]), float(xx[1]))
    #plt.ylim(float(xx[0]), float(xx[1]))
    
    # plt.axis('scaled')
    plt.grid()
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()
    

# k-fold cross-validation: i-th slice of data will be validation set, the rest will be training data
def get_kfold_data(k, i, X, y):
    """ i-th slice is validation data, union of {0,1, .., i-1, i+1, ...,k-1}-th slice is training data
    :k: > 1, int
    :i: 0, ..., k-1
    :X: data features
    :y: data labels
    :returns: a partition of full data
    """

    assert k > 1

    # // for floor function, returns int
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_validation, y_validation = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_validation, y_validation


# k-fold cross-validation training and error
def k_fold(k, X_train, y_train,
        num_epochs, learning_rate, weight_decay, batch_size):
    """ training k times on slices of data, each with num_epochs times
    """

    train_l_sum, valid_l_sum = 0, 0

    for i in range(k):

        # get the partition of data
        data = get_kfold_data(k, i, X_train, y_train)

        # create net
        net = get_nn()

        net.apply(init_weights) # recursively initializes all weights

        # train net
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)

        # sum errors from the last epoch (presumably the best error across all epochs)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')

        ## plot training and validation error when the first slice as validation data
        # if i == 0:
        #         d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
        #                  xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
        #                  legend=['train', 'valid'], yscale='log')

        if i==(k-1):
            show, close = True, True
        else:
            show, close = False, False
        plot(train_ls, valid_ls, show=show, close=close)

    # mean error over all all folds
    return train_l_sum / k, valid_l_sum / k

# hyperparameters
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

## train and compute error
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')

## train and predict
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_nn()

    net.apply(init_weights) # recursively initializes all weights

    # train without test data
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)

    # d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
    #          ylabel='log rmse', xlim=[1, num_epochs], yscale='log')

    # print('train_ls', train_ls)

    print(f'train log rmse {float(train_ls[-1]):f}')

    # Prediction: Apply the network to the test set, convert into numpy array
    preds = net(test_features).detach().numpy()

    ## Reformat it to export to Kaggle

    # append a column called `SalePrice` from a rank-1 numpy data
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])

    # two-column data with Id and SalePrice
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    # write to file
    filename = 'data/kaggle-housing/submission.csv'
    print('File written to', filename)
    submission.to_csv(filename, index=False)

    plot(train_ls, train_ls)

## run train and predict
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
