import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("./data/pima-indians-diabetes.data.csv", names=header)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)
print(rescaled_X)

# 테이터 분할
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.3)

# 모델 선택 및 분할
# model = LogisticRegression()
# model.fit(X_train, Y_train)
model = DecisionTreeClassifier(max_depth=100, min_samples_split=50, min_sample)

# 예측값 생성
y_pred = model.predic(X_test)

# 모델 정확도 계산
acc = accuracy_score(Y_test, y_pred)
print(acc)
