#!/usr/bin/env python3

import numpy as np
import pandas as pd

from common import prepare, sigmoid


weights_file = 'weights.csv'
weights = np.loadtxt(weights_file, delimiter=',')
print(f'loaded weights {weights} from CSV {weights_file}')

def main():
    test = pd.read_csv('csv/test.csv')
    test = prepare(test, with_id=True).to_numpy()

    submission = pd.DataFrame({
        'PassengerId': np.array([], dtype=np.int),
        'Survived': np.array([], dtype=np.int),
    })
    for i in range(len(test)):
        id = int(test[i][0])
        inpt = test[i][1:] # ignore id column
        # TODO: predict
        pred = sigmoid(inpt.dot(weights))
        submission = submission.append({
            'PassengerId': id,
            'Survived': pred
        }, ignore_index=True)

    submission['PassengerId'] = submission['PassengerId'].astype(int)
    submission['Survived'] =  submission['Survived'].round().fillna(0).astype(int)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
