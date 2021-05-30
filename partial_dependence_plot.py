import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

# データセットのロードと整理
dataset = load_wine()

x, t = dataset.data, dataset.target  # (178, 13) (178,)
columns = dataset.feature_names


# データの30%をテストデータにして分割
from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=2021)


# RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=2021)
model.fit(x_train, t_train)


# predict
y_pred = model.predict(x_test)

# accuracy
from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(t_test, y_pred)
print(f'Accuracy: {Accuracy}')
# model.score(x_test, t_test)

# Feature Importance (Gini Importance)
feature_importance = model.feature_importances_
importances = pd.DataFrame({"features": dataset.feature_names, "importances": feature_importance})
importances = importances.sort_values('importances', ascending=False)

plt.bar(height=importances['importances'].values, x=importances['features'])
plt.ylabel("Feature importances", fontsize=16, labelpad=20, weight='bold')
plt.xticks(rotation=75)

# Partial Dependence Plot
from sklearn.inspection import plot_partial_dependence
fig, (ax1, ax2, ax3) = plt.subplots(3, 4, figsize=(8, 8))
features = list(importances['features'].values)[:4]
df_x_train = pd.DataFrame(x_train, columns=columns)
plot_partial_dependence(model, df_x_train, features, target=0, ax=ax1)
plot_partial_dependence(model, df_x_train, features, target=1, ax=ax2, line_kw={"color": "red"})
plot_partial_dependence(model, df_x_train, features, target=2, ax=ax3, line_kw={"color": "green"})
# plt.show()

# predict
y_pred2 = model.predict(x_test)

# accuracy
from sklearn.metrics import accuracy_score
Accuracy_after_PDP = accuracy_score(t_test, y_pred2)
print(f'Accuracy after PDP: {Accuracy_after_PDP}')
