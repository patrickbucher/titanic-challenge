import pandas as pd


def prepare(orig, with_id=False):
    df = pd.DataFrame({
        'class': normalize(orig['Pclass']),
        'sex': orig['Sex'].map({'male': 0, 'female': 1}),
        'age': normalize(orig['Age']),
        'gentleman': ((orig['Pclass'] == 1) & (orig['Sex'] == 'male')).astype(int),
        'fare': normalize(orig['Fare']),
        'sibsp': normalize(orig['SibSp']),
        'parch': normalize(orig['Parch']),
        'cabin': orig['Cabin'].str.match('[ABC][0-9]+').fillna(False).astype(int),
        'wife': orig['Name'].str.contains('Mrs.').astype(int),
        #'embarked': orig['Embarked'].map({'S': 0, 'C': 1/2, 'Q': 1}),
    })

    # missing age: assume average age
    df['age'] = df['age'].fillna(df['age'].mean())

    #df['embarked'] = df['embarked'].fillna(0)

    if with_id:
        df['id'] = orig['PassengerId']
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

    return df

def normalize(v):
    return (v - v.mean()) / v.std()
