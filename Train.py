import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
import keras.backend as K

import numpy as np
import pickle
import math
import time

with open('./common/data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('./common/label.pkl', 'rb') as f:
    label = pickle.load(f)

matrixSize = len(data[0])  # 单个数据矩阵大小
datasetSize = len(data)  # 数据集大小
trainSize = math.floor(0.8 * datasetSize)  # 一部分作为训练，剩下的用于测试
batchSize = math.floor(0.2 * trainSize)  # 每批样本数
classesNum = 3  # 多少个种类
epochs = 50  # 训练次数
localtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # time

# x_data
xTrain = np.array(data[:trainSize])
xTest = np.array(data[trainSize:])
xTrain = keras.utils.normalize(xTrain, axis=1, order=2)
xTest = keras.utils.normalize(xTest, axis=1, order=2)

# y_data
yTrain = np.array(label[:trainSize])
yTest = np.array(label[trainSize:])
# 0 ~ classesNum 向量转矩阵
yTrain = keras.utils.to_categorical(yTrain, classesNum)
yTest = keras.utils.to_categorical(yTest, classesNum)

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(matrixSize,)))
model.add(Dropout(0.3))
model.add(Dense(classesNum, activation='softmax'))

model.summary()

tbCallBack = TensorBoard(
    log_dir='./logs/' + localtime,  # log 目录
    histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #   batchSize=32,     # 用多大量的数据计算直方图
    write_graph=True,  # 是否存储网络结构图
    write_grads=True,  # 是否可视化梯度直方图
    write_images=True,  # 是否可视化参数
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None
)

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    import keras.backend as K
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)


model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy', fmeasure, recall]
)

model.fit(
    xTrain,
    yTrain,
    batch_size=batchSize,
    epochs=epochs,
    verbose=1,
    validation_data=(xTest, yTest),
    callbacks=[tbCallBack]
)

# 预测测试数据准确度
score = model.evaluate(xTest, yTest)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model
model.save('./common/model.h5')
print('-----model saved-----')
