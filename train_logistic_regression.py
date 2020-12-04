#!/usr/bin/env python3

import numpy as np
import pandas as pd

from common import prepare, sigmoid


def cost(x, y, theta):
    z = x.dot(theta)
    c = -y * np.log(sigmoid(z)) - (1-y) * np.log((1 - sigmoid(z)))
    j = 1/len(x) * c.sum()
    return j


def main():
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare(train)
    print(train)

    goals = goals.to_numpy()
    inputs = train.to_numpy()

    theta = np.zeros((inputs.shape[1], 1))
    alpha = 0.01
    iterations = 100_000

    y = goals
    for i in range(iterations):
        m = len(inputs)
        x = inputs
        h = x.dot(theta)
        p = sigmoid(h)
        theta -= (alpha/m) * x.transpose().dot((p - y))
        c = cost(x, y, theta)
        if i % (iterations/10) == 0:
            print(f'cost={c:.3f}')


    predictions = sigmoid(x.dot(theta))
    predictions[:,0] = predictions[:,0].round()
    correct = len(inputs[(predictions == goals)[:,0]])
    
    accuracy = correct / len(goals) * 100
    print(f'accuracy: {accuracy:.3f}%')

    weights_file = 'weights.csv'
    np.savetxt('weights.csv', theta, delimiter=',')
    print(f'saved weights {theta} as CSV to {weights_file}')


if __name__ == '__main__':
    main()
