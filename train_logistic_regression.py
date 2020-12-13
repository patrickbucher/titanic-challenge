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
    data = pd.read_csv('csv/train.csv')

    # shuffle and split (0.7/0.3)
    data = data.sample(frac=1)
    m = len(data)
    m_train = round(m * 0.7)
    train = data.iloc[:m_train,]
    valid = data.iloc[m_train:,]

    goals_train = pd.DataFrame({'survived': train['Survived']})
    goals_valid = pd.DataFrame({'survived': valid['Survived']})
    train = prepare(train)
    valid = prepare(valid)

    goals_train = goals_train.to_numpy()
    goals_valid = goals_valid.to_numpy()
    inputs_train = train.to_numpy()
    inputs_valid = valid.to_numpy()

    theta = np.zeros((inputs_train.shape[1], 1))
    alpha = 0.01
    iterations = 100_000

    y = goals_train
    for i in range(iterations):
        m = len(inputs_train)
        x = inputs_train
        h = x.dot(theta)
        p = sigmoid(h)
        theta -= (alpha/m) * x.transpose().dot((p - y))
        c = cost(x, y, theta)
        if i % (iterations/10) == 0:
            print(f'cost={c:.3f}')


    predictions = sigmoid(inputs_valid.dot(theta))
    predictions[:,0] = predictions[:,0].round()
    correct = len(inputs_valid[(predictions == goals_valid)[:,0]])
    
    accuracy = correct / len(goals_valid) * 100
    print(f'accuracy: {accuracy:.3f}%')

    weights_file = 'weights.csv'
    np.savetxt('weights.csv', theta, delimiter=',')
    print(f'saved weights {theta} as CSV to {weights_file}')


if __name__ == '__main__':
    main()
