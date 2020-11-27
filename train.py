#!/usr/bin/env python3

import numpy as np
import pandas as pd

from common import prepare


def main():
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare(train)
    print(train)

    goals = goals.to_numpy()
    inputs = train.to_numpy()

    weights = np.zeros(inputs.shape[1])
    alpha = 1e-5

    for i in range(1000):
        for j in range(len(inputs)):
            goal = goals[j][0]
            inpt = inputs[j]

            pred = inpt.dot(weights)

            delta = goal - pred
            weigh_delta = delta * inpt
            adjustments = weigh_delta * alpha

            weights += adjustments

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
