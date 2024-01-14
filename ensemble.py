import pandas as pd

answer1 = pd.read_csv('./answer/answer20.csv')
answer2 = pd.read_csv('./answer/answer27.csv')
answer3 = pd.read_csv('./answer/answer45.csv')

tmp1 = 0.7*answer1.iloc[:, 1:]+0.3*answer2.iloc[:, 1:]
tmp2 = 0.7*tmp1+0.3*answer3.iloc[:, 1:]

ensemble = pd.concat([answer1['img_id'], tmp2], axis = 1)
ensemble.to_csv('./answer/ensemble.csv', index = False)