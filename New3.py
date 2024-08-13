import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("./data/pima-indians-diabetes.data.csv", names=header)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)

# 모델 선택 및 분할
model = LogisticRegression()

fold = KFold(n_splits=10, shuffle = True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring = 'accuracy')

# 평균 구하기 1
s = sum(acc)
l = len(acc)
avg = s / l

print(s, l, avg)