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
    train_y_list = train_label.tolist()  # list
    print(train_x_list)
    print(train_y_list)

    return train_x_list, train_y_list

def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


def peredata(content_data):
    print('step2: 分词处理及数据预处理将格式转换成tensor...')
    train_data = []  # 用于存储训练文档信息
    train_label = []  # 用于存储label数据（特征）
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    seg_content_list = [[] for index in range(len(content_data))]  # 存储每条content的分词结果
    count_word_frequency = {}  # 存储文档中出现的词及其对应的出现次数
    for i in range(len(content_data)):
        seg_content_data = jieba.cut(content_data[i][0])  # 默认是精确模式
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic) + 1
                count_word_frequency[word] = 1
            else:
                count_word_frequency[word] += 1
    if not os.path.exists('word_index_dict.json'):
        with open('word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)  # json.dumps在默认情况下，对于非ascii字符生成的是相对应的字符编码，而非原始字符
    print('文档中出现的词已经全部统计编码完，存放在word_index_dict.json 其中有词个数--->', len(word_index_dic))
    if not os.path.exists('count_word_frequency_dict.json'):
        with open('count_word_frequency_dict.json', 'w', encoding='utf-8') as f:
            json.dump(count_word_frequency, f, ensure_ascii=False)

    max_words = len(word_index_dic)  # 统计得到该文档用到的词的个数16895
    print('统计得到该文档用到的词的个数--->', max_words)
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
    # print('计算得到文本最大长度为-->',get_max_content_length(train_data))
    max_len = 100  # 计算得到文本最大长度为491
    max_words = 11525 +2  # 统计得到该文档用到的词的个数19307/20000
    epochs_num = 300
    batch_size_num = 64


    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)

    train_label = np.array(train_label)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, 16))
    model.add(keras.layers.LSTM(128, activation=tf.nn.tanh))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(95, activation=tf.nn.softmax))
    # model.add(keras.layers.Activation(tf.nn.softmax))
    model.summary()
    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[2000:2500]
    partial_x_train = train_data[0:2000]
    y_val = train_label[2000:2500]
    partial_y_train = train_label[0:2000]

    test_data = train_data[2500:]
    test_labels = train_label[2500:]

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
    sourcepath = os.path.abspath('../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    print('ss', dataPath)
    content, train_label = read_data(dataPath)
    print(train_label)
    clean_label(train_label)
    print('ssss', train_label)
    train_data = peredata(content)
    build_model(train_data, train_label)


# START
if __name__ == '__main__':
    execute()