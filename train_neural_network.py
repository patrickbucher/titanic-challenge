#!/usr/bin/env python3

import numpy as np
import pandas as pd

from common import prepare


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d1_sigmoid(z):
    sz = sigmoid(z)
    return sz * (1 - sz)


def relu(z):
    return np.maximum(0, z)


def d1_relu(z):
    mapper = lambda v: 0 if v < 0 else 1
    mapper = np.vectorize(mapper)
    return mapper(z)


def cost(x, y):
    m = x.shape[1]
    return -(1/m) * np.sum(y.dot(np.log(sigmoid(x).T)) + (1 - y).dot(np.log(1 - sigmoid(x).T)))


def main():

    # prepare test data
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare(train)
    x = train.to_numpy().T
    y = goals.to_numpy().T

    # initialize hyperparameters
    alpha = 0.001
    iters = int(1e3)
    batch = int(iters / 10)
    m = x.shape[1]
    n_x = x.shape[0]
    n_h = 10  # wild guess...
    n_y = 1

    # initialize parameters
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_h, n_h) * 0.01
    b2 = np.zeros((n_h, 1))
    w3 = np.random.randn(n_y, n_h) * 0.01
    b3 = np.zeros((n_y, 1))

    for i in range(iters):

        # forward propagation
        z1 = w1.dot(x) + b1
        a1 = relu(z1)
        z2 = w2.dot(a1) + b2
        a2 = relu(z2)
        z3 = w3.dot(a2) + b3
        a3 = sigmoid(z3)

        # calculate cost
        if i % batch == 0:
            j = cost(a3, y)
            print(j)

        # backward propagation
        da3 = - (np.divide(y, a3) - np.divide(1 - y, 1 - a3))
        dz3 = d1_sigmoid(da3)
        dw3 = (1/m) * dz3.dot(a2.T)
        db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)
        dz2 = w3.T.dot(dz3) * d1_relu(z2)
        dw2 = (1/m) * dz2.dot(a1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = w2.T.dot(dz2) * d1_relu(z1)
        dw1 = (1/m) * dz1.dot(x.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        w3 -= alpha * dw3
        b3 -= alpha * db3
        w2 -= alpha * dw2
        b2 -= alpha * db2
        w1 -= alpha * dw1
        b1 -= alpha * db1

    test = pd.read_csv('csv/test.csv')
    test = prepare(test, with_id=True)
    ids = test['id']
    x = test.to_numpy()[:,:n_x].T

    # prediction
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = relu(z2)
    z3 = w3.dot(a2) + b3
    a3 = sigmoid(z3)
    print(a3)

    submission = pd.DataFrame({
        'PassengerId': np.array(ids, dtype=np.int),
        'Survived': np.array(a3.T[:,0], dtype=np.int),
    })
    print(submission)



if __name__ == '__main__':
    main()
