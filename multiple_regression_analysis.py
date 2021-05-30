import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston

dataset = load_boston()

x, t = dataset.data, dataset.target  # (506, 13), (13, )
columns = dataset.feature_names

df = pd.DataFrame(x, columns=columns)
df['Terget'] = t

t = df['Terget'].values  # 目的変数 <class 'numpy.ndarray'>(.valuesによって)
x = df.drop(['Terget'], axis=1).values  # 説明変数 <class 'numpy.ndarray'>


from sklearn.model_selection import train_test_split

# データの30%をテストデータにして分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)


# モデルの定義(重回帰分析)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# モデルの学習
model.fit(x_train, t_train)

# 学習後のパラメータ(重み13個)
print(model.coef_)  # 入力変数に掛ける重みの値

# パラメータの大きさの分布
plt.figure(figsize=(10, 7))
plt.bar(x=columns, height=model.coef_)
# plt.show()

# 学習後のパラメータ(バイアス)
print(model.intercept_)

# モデルに対する決定係数(0<x<1 大きい数の方が良い)
print(f'train score:{model.score(x_train,t_train)}')  # trainに対する検証結果
print(f'test score:{model.score(x_test,t_test)}')
'''test > trainならば過学習の可能性大'''


# 推論
y = model.predict(x_test)
print(f'予測値: {y[0]}')
print(f'目標値: {t_test[0]}')


# 重回帰分析、ロジスティック回帰分析において、いくつかの説明変数間で線形関係（一次従属）が認められる場合、共線性があるといい、共線性が複数認められる場合は多重共線性があると言う。
# 完全な多重共線性が存在する場合、偏回帰係数を求めることができない。
# 多重共線性_PLS.pyへ →
