#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
# from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt
from ProcessData import process_data
import jieba
import json

# settings
max_len = 200
training_samples = 250
validation_samples = 30
max_words = 28918 + 2
embedding_dim = 16
epoch_num = 50
batch_size_num = 32


def peredata():
    '''
    将texts分词，向量化，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    train_data = []  # 用于存储训练文档信息

    filename = 'train'
    labels, texts = process_data(filename)
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    seg_content_list = [[] for index in range(len(labels))]  # 存储每条content的分词结果
    count_word_frequency = {}  # 存储文档中出现的词

    for i in range(len(texts)):
        seg_content_data = jieba.cut(texts[i])
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic) + 1
    # with open('word_index_dict.json', 'w', encoding='utf-8') as f:
    #     json.dump(word_index_dic, f, ensure_ascii=False)
    print('统计文档中词的出现个数', len(word_index_dic))
    for j in range(len(texts)):
        word_list = []
        for word in seg_content_list[j]:
            word_list.append(word_index_dic[word])
        train_data.append(word_list)
    print(len(train_data))
    print(labels)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)
    train_label = np.array(labels)
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)

    # shuffle
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]
    return train_data, train_label, word_index_dic


def parse_word_embedding(word_index):
    '''
    将预计算的词向量空间的word建立索引和矩阵
    :return:
    '''
    sgns_dir = 'G:\downloaddata\sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    embeddings_index = {}
    f = open(os.path.join(sgns_dir, 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def train_model(train_data, train_label, word_index):
    '''
    训练模型
    :return:训练时loss,acc
    '''
    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, embedding_dim, input_length=max_len))

    #MLP Model
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    # CNN Model
    model.add(keras.layers.Convolution1D(64, 3, padding='same'))
    model.add(keras.layers.MaxPool1D(6, 6, padding='same'))
    model.add(keras.layers.Convolution1D(32, 3, padding='same'))
    model.add(keras.layers.MaxPool1D(3, 3, padding='same'))
    model.add(keras.layers.Convolution1D(16, 3, padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())  # (批)规范化层
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    #LSTM Model
    # model.add(keras.layers.LSTM(128, activation=tf.nn.sigmoid))
    # model.add(keras.layers.Dropout(0.5))
    # # model.add(keras.layers.GlobalAveragePooling1D())
    # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    # 将sgns加载到模型中
    # embedding_matrix = parse_word_embedding(word_index)
    # model.layers[0].set_weights([embedding_matrix])
    # model.layers[0].trainable = False

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    x_train = train_data[:training_samples]
    y_train = train_label[:training_samples]
    x_val = train_data[training_samples: training_samples + validation_samples]
    y_val = train_label[training_samples: training_samples + validation_samples]

    test_data = train_data[200:]
    test_labels = train_label[200:]

    history = model.fit(x_train, y_train,
                        epochs=epoch_num,
                        batch_size=batch_size_num,
                        validation_data=(x_val, y_val))
    # # 保存评估MLP模型专用
    # model.save('pre_trained_MLPmodel_1.h5')
    # model.load_weights('pre_trained_MLPmodel_1.h5')

    # 保存评估CNN模型专用
    model.save('pre_trained_CNNmodel_1.h5')
    model.load_weights('pre_trained_CNNmodel_1.h5')

    # # 保存评估LSTM模型专用
    # model.save('pre_trained_LSTMmodel_1.h5')
    # model.load_weights('pre_trained_LSTMmodel_1.h5')

    test_acc = model.evaluate(test_data, test_labels)
    print('评估模型效果(损失-精度）：...', test_acc)

    print('开始绘图...')
    history_dict = history.history
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
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


if __name__ == '__main__':
    train_data, train_label, word_index_dic = peredata()
    train_model(train_data, train_label, word_index_dic)


    # model = keras.models.Sequential()
    # model.load_weights('pre_trained_model_1.h5')
    # test_acc = model.predict()
