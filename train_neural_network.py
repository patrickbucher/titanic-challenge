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
    mapper = lambda v: 0 if v <= 0 else 1
    mapper = np.vectorize(mapper)
    return mapper(z)


def cost(a, y):
    m = a.shape[1]
    return -(1/m) * np.sum(y.dot(np.log(a.T)) + (1 - y).dot(np.log(1 - a.T)))


def main():

    # prepare test data
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare(train)
    x = train.to_numpy().T
    y = goals.to_numpy().T

    # initialize hyperparameters
    alpha = 0.11
    iters = int(2e4)
    batch = int(iters / 10)
    m = x.shape[1]
    n_x = x.shape[0]
    n_h = 25  # wild guess...
    n_y = 1

    # initialize parameters
    #np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_h, n_h) * 0.01
    b2 = np.zeros((n_h, 1))
    w3 = np.random.randn(n_h, n_h) * 0.01
    b3 = np.zeros((n_h, 1))
    w4 = np.random.randn(n_y, n_h) * 0.01
    b4 = np.zeros((n_y, 1))

    for i in range(iters):

        # forward propagation
        z1 = w1.dot(x) + b1
        a1 = relu(z1)
        z2 = w2.dot(a1) + b2
        a2 = relu(z2)
        z3 = w3.dot(a2) + b3
        a3 = relu(z3)
        z4 = w4.dot(a3) + b4
        a4 = sigmoid(z4)

        # calculate cost
        if i % batch == 0:
            j = cost(a4, y)
            print(j)

        # backward propagation
        da4 = - (np.divide(y, a4) - np.divide(1 - y, 1 - a4))
        dz4 = da4 * d1_sigmoid(z4)
        dw4 = (1/m) * dz4.dot(a2.T)
        db4 = (1/m) * np.sum(dz4, axis=1, keepdims=True)

        da3 = w4.T.dot(dz4)
        dz3 = da3 * d1_relu(z3)
        dw3 = (1/m) * dz3.dot(a2.T)
        db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)

        da2 = w3.T.dot(dz3)
        dz2 = da2 * d1_relu(z2)
        dw2 = (1/m) * dz2.dot(a1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        da1 = w2.T.dot(dz2)
        dz1 = da1 * d1_relu(z1)
        dw1 = (1/m) * dz1.dot(x.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        w4 -= alpha * dw4
        b4 -= alpha * db4
        w3 -= alpha * dw3
        b3 -= alpha * db3
        w2 -= alpha * dw2
        b2 -= alpha * db2
        w1 -= alpha * dw1
        b1 -= alpha * db1

    test = pd.read_csv('csv/test.csv')
    test = prepare(test, with_id=True)
    ids = test['id']
    x = test.to_numpy()[:,1:].T

    # prediction
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = relu(z2)
    z3 = w3.dot(a2) + b3
    a3 = relu(z3)
    z4 = w4.dot(a3) + b4
    a4 = sigmoid(z4)
    y = np.around(a4)

    submission = pd.DataFrame({
        'PassengerId': np.array(ids, dtype=np.int),
        'Survived': np.array(y.T[:,0], dtype=np.int),
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
