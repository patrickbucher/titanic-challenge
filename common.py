import math

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + math.e ** -x)


def prepare(orig, with_id=False):
    df = pd.DataFrame({
        'class': normalize(orig['Pclass']),
        'sex': orig['Sex'].map({'male': 0, 'female': 1}),
        'age': normalize(orig['Age']),
        'wife': orig['Name'].str.contains('Mrs.').astype(int),
        'cabin': orig['Cabin'].str.match(r'\w+').fillna(False).astype(int),
        'sibsp': normalize(orig['SibSp']),
        'parch': normalize(orig['Parch']),
        'fare': normalize(orig['Fare']),
        'embarked': orig['Embarked'].map({
            'S': 0.336957,
            'C': 0.553571,
            'Q': 0.389610,
        }), # survival rates from features.py
    })

    # missing age: assume average age
    df['age'] = df['age'].fillna(df['age'].mean())

    age = orig['Age'].fillna(0)
    df['child'] = ((age < 16) & (age != 0)).astype(int)

    df['gentleman'] = ((df['sex'] == 0) & (orig['Pclass'] == 1)).astype(int)
    df['richbitch'] = ((df['sex'] == 1) & (orig['Pclass'] == 1)).astype(int)
    df['kiddo'] = ((df['parch'] > 0) & (df['child'])).astype(int)

    df['embarked'] = df['embarked'].fillna(0)

    def extract_ticket_number(t):
        space_at = t.find(' ')
        if space_at != -1:
            t = t[space_at:]
        try:
            t = int(t)
            return t
        except ValueError:
            return 0

    df['ticket'] = normalize(orig['Ticket'].apply(extract_ticket_number))

    if with_id:
        df['id'] = orig['PassengerId']
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

    return df


def normalize(v):
    return (v - v.mean()) / v.std()
