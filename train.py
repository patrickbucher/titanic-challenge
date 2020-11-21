#!/usr/bin/env python3

import numpy as np
import pandas as pd


def main():
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare_train(train)

    goals = goals.to_numpy()
    inputs = train.to_numpy()

    weights = np.array([1/4, 1/4, 1/4, 1/4])
    alpha = 1e-3

    for i in range(10_000):
        for j in range(len(inputs)):
            goal = goals[j][0]
            inpt = inputs[j]

            pred = inpt.dot(weights)

            delta = goal - pred
            weigh_delta = delta * inpt
            adjustments = weigh_delta * alpha

            weights += adjustments

    print('weights', weights)

    correct = 0
    for i in range(len(inputs)):
        inpt = inputs[j]
        pred = inpt.dot(weights)
        if round(pred) == goals[i]:
            correct += 1

    accuracy = correct / len(inputs) * 100
    print(f'accuracy: {accuracy:.3f}%')


def prepare_train(df):
    df = pd.DataFrame({
        'class': df['Pclass'] / max(df['Pclass']),
        'sex': df['Sex'].map({'male': 0, 'female': 1}),
        'age': df['Age'] / max(df['Age']),
        'fare': df['Fare'] / max(df['Fare']),
    })
    df['age'] = df['age'].fillna(df['age'].mean())
    return df


if __name__ == '__main__':
    main()
