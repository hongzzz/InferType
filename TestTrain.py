import keras
from keras.models import load_model
import numpy as np
import pickle
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

with open('common/data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('common/label.pkl', 'rb') as f:
    label = pickle.load(f)

sampleSize = len(data[0])  # 单个数据矩阵大小
dataSetSize = len(data)  # 数据集大小

xTest = np.array(data)
yTest = np.array(label)
yTest = keras.utils.to_categorical(yTest, 3)

print('-----load model-----')
model = load_model('./common/model.h5')
model.summary()

score = model.evaluate(xTest, yTest)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

p = model.predict_classes(xTest)
print(p[0])