#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
from nltk import tokenize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, RepeatVector, Permute, Multiply, Lambda, BatchNormalization, LeakyReLU
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, CuDNNLSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint

from keras.optimizers import Adam, Adadelta, SGD


from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

# from tqdm import tqdm
import math
from matplotlib import pyplot as plt
import seaborn as sn
# from colored import fg, bg, attr

class AttentionLayer(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(input_shape[-1], 1),
                                 initializer=self.init,
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

        # RNN 구조를 거쳐서 나온 hidden states (x)에 single layer perceptron (tanh activation)
        # 적용하여 나온 벡터가 uit
        u_it = K.tanh(K.dot(x, self.W) + self.b)

        # uit와 uw (혹은 us) 간의 similarity를 attention으로 사용
        # softmax를 통해 attention 값을 확률 분포로 만듬
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)

        return a_it

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 40

# 1. create data
signal_data = np.cos(np.arange(1600)*(20*np.pi/1000))[:,None]

# process data
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# data split
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)
print('#############',x_train.shape, y_train.shape)
print(x_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print('@@@@@@@@@@@@@@@@@@@',x_train.shape, y_train.shape)

lstm_dim = 50

input_data = Input(shape=(40,1))
bilstm = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_data)  # lstm입력은 (N, X, Y) 3차원이어여한다
# bilstm = LSTM(2*lstm_dim, return_sequences=True)(input_data)  # lstm입력은 (N, X, Y) 3차원이어여한다
# bilstm_output = Dense(1)(bilstm)

attention_layer = AttentionLayer()(bilstm)
print(attention_layer)

repeated_word_attention = RepeatVector(lstm_dim * 2)(attention_layer)
repeated_word_attention = Permute([2, 1])(repeated_word_attention)
sentence_representation = Multiply()([bilstm, repeated_word_attention])
sentence_representation = Lambda(lambda x: K.sum(x, axis=1))(sentence_representation)

bilstm_output = Dense(1)(sentence_representation)

model = Model(inputs=[input_data],
            outputs=[bilstm_output])
# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val))

model.summary()


# 5. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, verbose=0)
model.reset_states()
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
plt.legend()
plt.show()

# Attention check
# 위의 attention layer 부분을 output으로
attention_extractor = Model(inputs=[input_data],
                             outputs=[attention_layer])
# (N, 40)
attention_expamle = attention_extractor.predict(x_test)


print(attention_expamle[0])

plt.bar(np.arange(attention_expamle.shape[1]), attention_expamle[2])
plt.show()