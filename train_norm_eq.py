#!/usr/bin/env python3

import numpy as np
import pandas as pd

from common import prepare


def main():
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare(train)

    goals = goals.to_numpy()
    inputs = train.to_numpy()

    # add column x0 := 1
    (rows, cols) = inputs.shape
    X = np.ones((rows, cols+1))
    X[:, 1:] = inputs

    # normal equation
    Xt = X.transpose()
    theta = np.linalg.inv(Xt.dot(X)).dot(Xt.dot(goals))

    # get rid of x0 again
    weights = theta.transpose()[0][1:]

    correct = 0
    for i in range(len(inputs)):
        inpt = inputs[i]
        pred = inpt.dot(weights).round().astype(int)
        goal = goals[i][0]
        if pred == goal:
            correct += 1

    accuracy = correct / len(goals) * 100
    print(f'accuracy: {accuracy:.3f}%')

    weights_file = 'weights.csv'
    np.savetxt('weights.csv', weights, delimiter=',')
    print(f'saved weights {weights} as CSV to {weights_file}')


if __name__ == '__main__':
    main()
