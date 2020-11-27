#!/usr/bin/env python3

import pandas as pd

from common import prepare


def main():
    data = pd.read_csv('csv/train.csv')
    df = data.groupby('Embarked').aggregate({
        'Survived': 'sum',
        'PassengerId': 'count',
    })
    df = df.rename(columns={
        'Survived': 'survived',
        'PassengerId': 'passengers'
    })
    df['survival_rate'] = df['survived'] / df['passengers']
    print(df)

if __name__ == '__main__':
    main()
