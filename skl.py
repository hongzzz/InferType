import numpy as np
import pickle
import math
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

with open('./common/data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('./common/label.pkl', 'rb') as f:
    label = pickle.load(f)

def calculate_result(actual,pred):
    # 输出各评价指标
    accuracy = metrics.accuracy_score(actual,pred)
    recallScore = metrics.recall_score(actual,pred, average='weighted')
    f1Score = metrics.f1_score(actual,pred, average='weighted')
    print("Accuracy:{0:.3f}".format(accuracy))
    print("Recall Score:{0:.3f}".format(recallScore))
    print("F1Score:{0:.3f}".format(f1Score))


x = np.array(data)
y = np.array(label)

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.3, random_state=1251)

# 标准化
scaler = MaxAbsScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)

# 训练并评估
svc = SVC(kernel='rbf')
svc.fit(xTrain, yTrain)
yPred = svc.predict(xTest)

# 输出评估结果
calculate_result(yTest, yPred)


