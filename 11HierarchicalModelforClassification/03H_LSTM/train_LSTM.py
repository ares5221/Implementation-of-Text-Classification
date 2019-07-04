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
from gensim.models import word2vec
from ProcessData import process_data

def read_data(select):

    train_data = []  # 用于存储训练文档信息
    texts, labels, classes_num = process_data(select)
    tmp_data = np.array(texts)
    tmp_label = np.array(labels)
    indices = np.arange(len(labels))  # shuffle
    np.random.shuffle(indices)
    tmp_data = tmp_data[indices]
    tmp_label = tmp_label[indices]

    train_data = tmp_data.tolist()
    train_label = tmp_label.tolist()
    return train_data, train_label

def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


def peredata(content_data):
    print('step2: 分词处理及数据预处理将格式转换成tensor...')
    train_data = []  # 用于存储训练文档信息
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    seg_content_list = [[] for index in range(len(content_data))]  # 存储每条content的分词结果
    for i in range(len(content_data)):
        seg_content_data = jieba.cut(content_data[i][0])  # 默认是精确模式
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic)
    if not os.path.exists('word_index_dict.json'):
        with open('word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)  # json.dumps在默认情况下，对于非ascii字符生成的是相对应的字符编码，而非原始字符
    print('文档中出现的词已经全部统计编码完，存放在word_index_dict.json 其中有词个数--->', len(word_index_dic))
    for j in range(len(seg_content_list)):
        word_list = []
        for word in seg_content_list[j]:
            word_list.append(word_index_dic[word])
        train_data.append(word_list)
    if not os.path.exists('train_data.txt'):
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
    print('step4: start build model...')
    print('计算得到文本最大长度为-->', get_max_content_length(train_data))
    max_len = 100
    max_words = 11166 + 1
    epochs_num = 500
    batch_size_num = 64
    embedding_dim = 16

    train_data = keras.preprocessing.sequence.pad_sequences(train_data, padding='post', maxlen=max_len)
    train_label = np.array(train_label)
    print(train_data.shape)
    print(train_label.shape)

    print('step5:导入 outer embeding')
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
    print('setp4: start building the lstm model...')

    embedding_dim = 300
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = [0 for i in range(300)]
    print('step5:导入 outer embeding完成')

    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, embedding_dim))
    model.add(keras.layers.LSTM(128, activation=tf.nn.tanh))
    # model.add(keras.layers.Dropout(0.2)
    # )
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(20, activation=tf.nn.sigmoid))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[2100:]
    partial_x_train = train_data[0:2100]
    y_val = train_label[2100:]
    partial_y_train = train_label[0:2100]

    test_data = train_data[2200:]
    test_labels = train_label[2200:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs_num,
                        batch_size=batch_size_num,
                        validation_data=(x_val, y_val)
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


# setp 0
def execute():
    select = 'all'
    content, train_label = read_data(select)
    train_data, word_index = peredata(content)
    build_model(train_data, train_label, word_index)


# START
if __name__ == '__main__':
    execute()
