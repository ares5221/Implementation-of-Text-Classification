#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from ProcessData import process_data
from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

'''
获取标注数据
将标注数据随机分成训练数据与测试数据8：2
全部转换为bert encoding向量
通过MLP分类
'''


def peredata(select):
    '''
    将句子通过BERT embedding转为vector，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    train_data = []  # 用于存储训练文档信息
    texts, labels, classes_num = process_data(select)
    tmp_data = np.array(texts)
    tmp_label = np.array(labels)
    indices = np.arange(len(labels))  # shuffle
    np.random.shuffle(indices)
    tmp_data = tmp_data[indices]
    tmp_label = tmp_label[indices]

    texts = tmp_data.tolist()
    labels = tmp_label.tolist()
    print(texts)
    print(labels)
    bc = BertClient()
    for i in range(len(texts)):
        v1 = bc.encode(texts[i])
        train_data.append(v1[0])
    train_label = np.array(labels)

    if not os.path.exists(select + "_train_data.npy"):  # save data
        np.save(select + "_train_data.npy", train_data)
    if not os.path.exists(select + "_train_label.npy"):
        np.save(select + "_train_label.npy", train_label)

    print('step2: BERT转换向量完成，存储在_train_data.npy, _train_label.npy')
    return train_data, train_label, classes_num


def GetTrainData(select):
    train_data = np.load(select + "_train_data.npy")
    train_label = np.load(select + "_train_label.npy")
    print(select, '直接导入分类数据成功')
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    indices = np.arange(train_data.shape[0])  # shuffle
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]

    set_label = set(train_label)
    classes_num = len(set_label)
    print('############', set_label, classes_num)
    return train_data, train_label, classes_num


def train_MLPmodel(train_data, train_label, classes_num, select):
    print('step3: start build MLP model...')
    max_len = 768  # BERT Embedding length
    epochs_num = 2000
    batch_size_num = 64
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)
    train_label = np.array(train_label)
    train_label = np_utils.to_categorical(train_label, classes_num)

    model = keras.Sequential()
    # model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(768,)))
    # model.add(keras.layers.Dense(classes_num, activation=tf.nn.softmax))
    model.add(keras.layers.Dense(classes_num, activation=tf.nn.softmax, input_shape=(768,)))
    model.summary()
    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    partial_x_train, x_val, partial_y_train, y_val = train_test_split(train_data, train_label, test_size=0.3)
    test_data = train_data[:]
    test_labels = train_label[:]

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
    tt1 = select + ' :Training and validation loss'
    plt.title(tt1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()  # clear figure
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    tt2 = select + ' :Training and validation accuracy'
    plt.title(tt2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print('模型训练结束！！！！！')


if __name__ == '__main__':
    print('采用BERT+MLP模型对标注文本按一级标签做分类')
    option = ['all']
    for i in range(0, len(option)):
        select = option[i]
        if not os.path.exists(select + "_train_vec.npy"):
            train_data, train_label, classes_num = peredata(select)
        else:
            train_data, train_label, classes_num = GetTrainData(select)  # 获取训练数据
        train_MLPmodel(train_data, train_label, classes_num, select)
