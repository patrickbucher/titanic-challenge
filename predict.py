#!/usr/bin/env python3

import numpy as np
import pandas as pd


weights = np.array([0.01200623, 0.56444341, 0.24316459, 0.9375338])

def main():
    test = pd.read_csv('csv/test.csv')
    test = prepare_test(test).to_numpy()

    submission = pd.DataFrame({
        'PassengerId': np.array([], dtype=np.int),
        'Survived': np.array([], dtype=np.int),
    })
    for i in range(len(test)):
        id = int(test[i][0])
        inpt = test[i][1:] # ignore id column
        pred = inpt.dot(weights)
        submission = submission.append({
            'PassengerId': id,
            'Survived': pred
        }, ignore_index=True)

    submission['PassengerId'] = submission['PassengerId'].astype(int)
    submission['Survived'] =  submission['Survived'].round().fillna(0).astype(int)
    submission.to_csv('submission.csv', index=False)


def prepare_test(df):
    df = pd.DataFrame({
        'id': df['PassengerId'],
        'class': df['Pclass'] / max(df['Pclass']),
        'sex': df['Sex'].map({'male': 0, 'female': 1}),
        'age': df['Age'].fillna(df['Age'].mean()) / max(df['Age']),
        'fare': df['Fare'] / max(df['Fare']),
    })
    return df

if __name__ == '__main__':
    main()
