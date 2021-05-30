import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/mamiyariku/Desktop/program_lesson/教師あり学習の練習/regression_pls.csv')  # (1290, 197)

t = df['Target'].values  # (1290,)
x = df.drop('Target', axis=1).values  # (1290, 196)

from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, t_train)

print(f'train score:{model.score(x_train,t_train)}')
print(f'test score:{model.score(x_test,t_test)}')


# 相関関係を調べる
df_corr = df.corr()

# 相関係数のヒートマップ
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.iloc[:20, :20], annot=True)
# plt.show()

sns.jointplot(x='x1', y='x16', data=df)
# plt.show()

'''
重回帰分析、ロジスティック回帰分析において、いくつかの説明変数間で線形関係（一次従属）が認められる場合、共線性があるといい、共線性が複数認められる場合は多重共線性があると言う。
完全な多重共線性が存在する場合、偏回帰係数を求めることができない。
'''


# PLS(Partial Least Squares)の実装
# 入力値を目的変数の情報と併せて新たな変数(潜在変数)を作成(主成分分析のようなもの)し重回帰分析をする

from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=11)  # n_components(ハイパーパラメータ):変換後の次元数

pls.fit(x_train, t_train)

print(f'train score:{pls.score(x_train,t_train)}')
print(f'test score:{pls.score(x_test,t_test)}')
