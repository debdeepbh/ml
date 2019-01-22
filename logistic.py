# logistic regression from https://github.com/taspinar/siml/blob/master/notebooks/Linear%20Regression%2C%20Logistic%20Regression.ipynb
import random
import numpy as np
import pandas as pd


def func2 (x_i):
    if x_i[1] + x_i[2] >= 13: # if studying time + sleeping time >=13
        return 0                # student fails
    else:
        return 1                # otherwise passes

def generate_data2 (num_points):
    X = np.zeros(shape=(num_points, 3))  # a Nx3 matrix of zeroes
    Y = np.zeros(shape=num_points)       # a vector of length num_points
    for ii in range(num_points):
        X[ii][0] = 1;                   # what this, probably useless data
        X[ii][1] = random.random()*9 + 0.5  # studying data
        X[ii][2] = random.random()*9 + 0.5  # sleeping data
        Y[ii] = func2(X[ii])         # generate pass-fail data, based on rule
    return X, Y                         # return dependent and independent variables

X, Y = generate_data2 (300)              # generate 300 data points


class LogisticRegression():
    " Class for performing logistic regression"
    def __init__(self, learning_rate = 0.7, max_iter = 1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.theta = []
        self.no_examples  = 0
        self.no_features = 0
        self.X = None
        self.Y = None

    def add_bias_col(self, X):
        bias_col = np.ones((X.shape[0], 1))
        return np.concatenate([bias_col, X], axis =1)
    def hypothesis(self, X):
        # the hypothesis is linear here
        # for quadratic hypothesis, take theta_i to be the coefficient of
        # terms 1, x_i^2, x_i x_j and x_i
        return 1 / (1 + np.exp(-1.0 * np.dot(X, self.theta)))

    def cost_function(self):
        """
        We will use the binary cross entropy as the cost function. https://en.wikipedia.org/wiki/Cross_entropy
        """
        predicted_Y_values = self.hypothesis(self.X)
        cost = (-1.0/self.no_examples) * np.sum(self.Y * np.log(predicted_Y_values) + (1 - self.Y) * (np.log(1-predicted_Y_values)))
        return cost
    def gradient(self):
        predicted_Y_values = self.hypothesis(self.X)
        grad = (-1.0/self.no_examples) * np.dot((self.Y-predicted_Y_values), self.X)
        return grad

    def gradient_descent(self):
        for iter in range(1,self.max_iter):
            cost = self.cost_function()
            delta = self.gradient()
            self.theta = self.theta - self.learning_rate * delta
            if iter % 100 == 0: print("iteration %s : cost %s " % (iter, cost))

    def train(self, X, Y):
        self.X = self.add_bias_col(X)
        self.Y = Y
        self.no_examples, self.no_features = np.shape(X)
        self.theta = np.ones(self.no_features + 1)
        self.gradient_descent()

    def classify(self, X):
        X = self.add_bias_col(X)
        predicted_Y = self.hypothesis(X)
        predicted_Y_binary = np.round(predicted_Y)
        #return predicted_Y_binary
        return predicted_Y


## testing with real data

# load the csv file
df = pd.read_csv('datasets/iris.data', header = None)

# choose a sample of proportion 0.7 to train the model
df_train = df.sample(frac = 0.7)

# get the independent and dependent variables
### !!!!!!!!!! strange!!! 0:4 means 0,1,2,3. 4 is not included!
# get the independent variables
X_train = df_train.values[:,0:4].astype(float)
# get the dependent variable
Y_train = df_train.values[:,4]

####################
# create a dictionary to assign numbers to names
flower2name = { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 }

to_bin_y = { 1: { 'Iris-setosa': 1, 'Iris-versicolor': 0, 'Iris-virginica': 0 },
             2: { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 0 },
             3: { 'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1 }
             }
YY_train = np.array([to_bin_y[3][x] for x in Y_train])
Y2_train = np.array([to_bin_y[2][x] for x in Y_train])
Y1_train = np.array([to_bin_y[1][x] for x in Y_train])


print(YY_train)

# test how dictionary works
YM_train = np.array([flower2name[k] for k in Y_train])
print(YM_train)
# training 3 at the smae time does not work
# have to assign high prob-1 and low-prob 0 to the y-vals to train


# creating a new instance of the class
lr_3 = LogisticRegression()
lr_2 = LogisticRegression()
lr_1 = LogisticRegression()

# This only trains the 3rd kind of flower
lr_3.train(X_train, YY_train)
lr_2.train(X_train, Y2_train)
lr_1.train(X_train, Y1_train)

# test data
df_test = df.loc[~df.index.isin(df_train.index)]
X_test = df_test.values[:,0:4].astype(float)
Y_test = [flower2name[k] for k in df_test.values[:,4]]

#print(lr_3.classify(X_test))
#print(lr_2.classify(X_test))
#print(lr_1.classify(X_test))

out = np.matrix([lr_1.classify(X_test), lr_2.classify(X_test), lr_3.classify(X_test)])
#print(out)

testNums = np.argmax(out, axis = 0)
print(testNums)
print(np.array(Y_test))
print(testNums == Y_test)

#print(df_train.values[:,4])
#f2n = np.array([flower2name[k] for k in df.values[:,4]])
#print(f2n)
#df2 = df
#df2[5] = f2n
#df2.drop(columns = 4)
#print(df2)
#df2.to_csv('datasets/irisnum.data', sep=',')
