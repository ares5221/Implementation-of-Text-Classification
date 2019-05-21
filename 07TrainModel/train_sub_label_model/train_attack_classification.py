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


def read_data(datapath):
    df = pd.read_csv(datapath, encoding='utf-8')# pandas读取csv数据
    print(df.shape)#查看数据大小
    print(df.head())#查看前5行
    # print(df['attack_label'])
    # print(df['attack_content'])
    X = df['attack_content']
    Y = df['attack_label']
    train_data = np.array(X)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    train_label = np.array(Y)  # np.ndarray()
    train_y_list = train_label.tolist()  # list
    # print(train_x_list)
    # print(train_y_list)
    return train_x_list, train_y_list


def peredata(content_data):
    print('step2: 分词处理及数据预处理将格式转换成tensor...')
    train_data = []  # 用于存储训练文档信息
    train_label = []  # 用于存储label数据（特征）
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    seg_content_list = [[] for index in range(len(content_data))]  # 存储每条content的分词结果
    for i in range(len(content_data)):
        seg_content_data = jieba.cut(content_data[i])  # 默认是精确模式
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic) + 1

    if not os.path.exists('word_index_dict.json'):
        with open('word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)  # json.dumps在默认情况下，对于非ascii字符生成的是相对应的字符编码，而非原始字符
    print('文档中出现的词已经全部统计编码完，存放在word_index_dict.json 其中有词个数--->', len(word_index_dic))
    for j in range(len(seg_content_list)):
        word_list = []
        for word in seg_content_list[j]:
            word_list.append(word_index_dic[word])
        train_data.append(word_list)
    if not os.path.exists('train_data.txt'):# 将每条文本的编码后矩阵保存在train_data.txt
        save_traindata_txt(train_data)
    return train_data


def save_traindata_txt(train_data):
    print('将traindata保存在txt文件中')
    file = open('train_data.txt', 'w')
    for datalist in train_data:
        # print(datalist, 'xxxxx')
        file.write(str(datalist))
        file.write('\n')
    file.close()


def get_max_content_length(data_list):
    max_len, index = 0, 0
    for i in range(len(data_list)):
        # print(data_list[i])
        if max_len < len(data_list[i]):
            max_len = len(data_list[i])
            index = i
    # print('content中最长的文本长度为：', max_len, 'index：', index)
    return max_len


def build_model(train_data,train_label):
    print('step4: start build model...')
    print('计算得到文本最大长度为-->',get_max_content_length(train_data))
    max_len = 20  # 计算得到文本最大长度为301
    max_words = 2582 + 2  # 统计得到该文档用到的词的个数19307/20000
    epochs_num = 300
    batch_size_num = 32


    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)

    train_label = np.array(train_label)
    print(train_label)
    print(train_data[0])
    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, 16))
    model.add(keras.layers.LSTM(128, activation=tf.nn.tanh))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(3, activation=tf.nn.softmax))
    # model.add(keras.layers.Activation(tf.nn.softmax))
    model.summary()
    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[150:180]
    partial_x_train = train_data[0:150]
    y_val = train_label[150:180]
    partial_y_train = train_label[0:150]

    test_data = train_data[150:]
    test_labels = train_label[150:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs_num,
                        batch_size=batch_size_num,
                        validation_data=(x_val, y_val),
                        shuffle=True
                        )

    results = model.evaluate(test_data, test_labels)
    print('step5: 评估模型效果(损失-精度）：...', results)
    filepath = 'model-LSTM.h5'
    model.save_weights(filepath)
    # # 将模型权重保存到指定路径，文件类型是HDF5（后缀是.h5）
    #
    # model.load_weights(filepath, by_name=False)
    # # 从HDF5文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。
    # # 如果想将权重载入不同的模型（有些层相同）中，则设置by_name=True，只有名字匹配的层才会载入权重


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
        if data[i] == 3:
            data[i] = 0

# setp 0
def execute():
    real_file = 'attack_data.csv'
    content, train_label = read_data(real_file)
    print('old label-->',train_label)
    clean_label(train_label)
    print('new label-->',train_label)
    train_data = peredata(content)
    build_model(train_data, train_label)


# START
if __name__ == '__main__':
    execute()

