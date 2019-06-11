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


#采用demo的网络结构做多分类，
def read_data(datapath):
    df = pd.read_csv(datapath, encoding='utf-8')  # pandas读取csv数据
    # print(df.head())#查看前5行
    make_label(df)
    X = df[['Content']]
    Y = df.multilabel
    train_data = np.array(X)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    train_label = np.array(Y)  # np.ndarray()
    train_y_list = train_label.tolist()  # list
    # print(train_x_list)
    # print(train_y_list)
    print('step1： 读取文本及对应label success，数据格式：', df.shape)# 查看数据大小
    return train_x_list, train_y_list


def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


def peredata(content_data):
    print('step2: 分词处理及数据预处理将格式转换成tensor...')
    train_data = []  # 用于存储训练文档信息
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    seg_content_list = [[] for index in range(len(content_data))]  # 存储每条content的分词结果
    for i in range(len(content_data)):
        seg_content_data = jieba.cut(content_data[i][0])
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic) + 1
    if not os.path.exists('word_index_dict.json'):
        with open('word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)
    print('setp2: 文档中出现的词已经全部统计编码完，存放在word_index_dict.json 其中有词个数--->', len(word_index_dic))

    for j in range(len(seg_content_list)):
        word_list = []
        for word in seg_content_list[j]:
            word_list.append(word_index_dic[word])
        train_data.append(word_list)
    if not os.path.exists('train_data.txt'):  # 将每条文本的编码后矩阵保存在train_data.txt
        save_traindata_txt(train_data)
    return train_data, word_index_dic


def save_traindata_txt(train_data):
    print('将traindata保存在txt文件中')
    file = open('train_data.txt', 'w')
    for datalist in train_data:
        file.write(str(datalist))
        file.write('\n')
    file.close()


def get_max_content_length(data_list):
    max_len, index = 0, 0
    for i in range(len(data_list)):
        if max_len < len(data_list[i]):
            max_len = len(data_list[i])
            index = i
    # print('content中最长的文本长度为：', max_len, 'index：', index)
    return max_len


def build_model(train_data, train_label, word_index):
    print('setp3: start building the demo model...')
    # print('计算得到文本最大长度为-->',get_max_content_length(train_data))
    max_len = 200  # 计算得到文本最大长度为491
    max_words = 11525 + 1
    epochs_num = 3000
    batch_size_num = 64

    train_data = keras.preprocessing.sequence.pad_sequences(train_data, padding='post', maxlen=max_len)
    train_label = np.array(train_label)

    print('step4:导入 outer embeding')
    sgns_dir = r'G:\downloaddata\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    embeddings_index = {}
    f = open(os.path.join(sgns_dir, 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_dim = 300
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = [0 for i in range(300)]
    print('step4:导入 outer embeding完成')

    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(95, activation=tf.nn.sigmoid))
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
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
    sourcepath = os.path.abspath('../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    content, train_label = read_data(dataPath)
    clean_label(train_label)
    train_data, word_index = peredata(content)
    build_model(train_data, train_label, word_index)


# START
if __name__ == '__main__':
    execute()
