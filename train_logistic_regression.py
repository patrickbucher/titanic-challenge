#!/usr/bin/env python3

import numpy as np
import pandas as pd

from common import prepare, sigmoid


def main():
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare(train)
    print(train)

    goals = goals.to_numpy()
    inputs = train.to_numpy()

    theta = np.zeros((inputs.shape[1], 1))
    alpha = 1e-5

    y = goals
    for i in range(1000):
        m = len(inputs)
        x = inputs
        h = theta.transpose().dot(x.transpose())
        p = sigmoid(h.transpose())
        theta -= alpha/m * x.transpose().dot((p - y))

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
