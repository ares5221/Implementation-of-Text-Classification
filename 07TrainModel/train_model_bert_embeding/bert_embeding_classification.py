#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import pandas as pd
import jieba
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from bert_serving.client import BertClient

def read_data(datapath):
    df = pd.read_csv(datapath, encoding='utf-8')# pandas读取csv数据
    print(df.shape)#查看数据大小
    # print(df.head())#查看前5行
    # print(df['Label'])
    # print(df['Content'])
    make_label(df)
    # print(df.head())
    # print(df['multilabel'])
    X = df[['Content']]
    Y = df.multilabel
    # print(X,Y)
    train_data = np.array(X)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    train_label = np.array(Y)  # np.ndarray()
    np.save('Y_data.npy', train_label)
    print(train_label.shape)
    train_y_list = train_label.tolist()  # list
    print(train_x_list)
    print(train_y_list)

    return train_x_list, train_y_list

def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))

# 通过bert将句子embeding
def peredata(content_data):
    print('step2: 分词处理及数据预处理将格式转换成tensor...')
    train_data = []  # 用于存储训练文档信息
    train_label = []  # 用于存储label数据（特征）
    bc = BertClient()
    for i in range(len(content_data)):
        print(content_data[i])
        v1 = bc.encode(content_data[i])
        print(v1[0])
        train_data.append(v1[0])
    if not os.path.exists('train_data.txt'):# 将每条文本的编码后矩阵保存在train_data.txt
        save_traindata_txt(train_data)
    np.save('X_data.npy', train_data)
    return train_data


def save_traindata_txt(train_data):
    print('将traindata保存在txt文件中')
    file = open('train_data.txt', 'w')
    for datalist in train_data:
        file.write(str(datalist))
        file.write('\n')
    file.close()


def build_model(train_data,train_label):
    print('step4: start build model...')

    # print('计算得到文本最大长度为-->',get_max_content_length(train_data))
    max_len = 100  # 计算得到文本最大长度为491
    max_words = 11525 +2  # 统计得到该文档用到的词的个数19307/20000
    epochs_num = 300
    batch_size_num = 64

    #
    # train_data = keras.preprocessing.sequence.pad_sequences(train_data,
    #                                                         padding='post',
    #                                                         maxlen=max_len)
    #
    # train_label = np.array(train_label)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(768, 16))
    # model.add(keras.layers.LSTM(95,activation='relu',input_shape=(2000,758)))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.GlobalAveragePooling1D())
    # model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(95, activation=tf.nn.softmax))
    model.summary()
    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[2500:2800]
    partial_x_train = train_data[0:2500]
    y_val = train_label[2500:2800]
    partial_y_train = train_label[0:2500]

    test_data = train_data[2800:]
    test_labels = train_label[2800:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs_num,
                        batch_size=batch_size_num,
                        validation_data=(x_val, y_val),
                        shuffle=True
                        )

    results = model.evaluate(test_data, test_labels)
    print('step5: 评估模型效果(损失-精度）：...', results)

    print('step6: 开始绘图...')
    history_dict = history.history
    print(history.history)
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()  # clear figure
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print('模型训练结束！！！！！')



def clean_label(data):
    for i in range(len(data)):
        if data[i] == 95:
            data[i] = 0

# setp 0
def execute():
    sourcepath = os.path.abspath('../../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    print('ss', dataPath)
    if not os.path.exists('X_data.npy'):
        content, train_label = read_data(dataPath)
        print(train_label)
        clean_label(train_label)
        print('ssss', train_label)
        train_data = peredata(content)
    X = np.load('X_data.npy')
    Y = np.load('Y_data.npy')
    print(X.shape)
    print(Y.shape)
    build_model(X, Y)


# START
if __name__ == '__main__':
    execute()