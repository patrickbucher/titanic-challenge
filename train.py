#!/usr/bin/env python3

import numpy as np
import pandas as pd


def main():
    train = pd.read_csv('csv/train.csv')
    goals = pd.DataFrame({'survived': train['Survived']})
    train = prepare_train(train)
    print(train)

    goals = goals.to_numpy()
    inputs = train.to_numpy()

    weights = np.zeros(8)
    alpha = 1e-3

    for i in range(100):
        for j in range(len(inputs)):
            goal = goals[j][0]
            inpt = inputs[j]

            pred = inpt.dot(weights)

            delta = goal - pred
            weigh_delta = delta * inpt
            adjustments = weigh_delta * alpha

            weights += adjustments

    weight_strings = [f'{w:.6f}' for w in weights.tolist()]
    weight_str = ', '.join(weight_strings)
    print(f'weights = np.array([{weight_str}])')

    correct = 0
    for i in range(len(inputs)):
        inpt = inputs[i]
        pred = inpt.dot(weights).round().astype(int)
        goal = goals[i][0]
        if pred == goal:
            correct += 1

    accuracy = correct / len(goals) * 100
    print(f'accuracy: {accuracy:.3f}%')


def prepare_train(df):
    df = pd.DataFrame({
        'class': 1 - df['Pclass'] / max(df['Pclass']),
        'sex': df['Sex'].map({'male': 0, 'female': 1}),
        'age': 1 - df['Age'] / max(df['Age']),
        'fare': df['Fare'] / max(df['Fare']),
        'sibsp': df['SibSp'] / max(df['SibSp']),
        'parch': df['Parch'] / max(df['Parch']),
        'wife': df['Name'].str.contains('Mrs.').astype(int),
        'cabin': df['Cabin'].str.match('[ABC][0-9]+').fillna(False).astype(int),
    })
    df['age'] = df['age'].fillna(df['age'].mean())
    return df


if __name__ == '__main__':
    main()
