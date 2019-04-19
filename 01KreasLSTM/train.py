#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt
from ProcessData import process_data
import jieba

# settings
max_len = 100
training_samples = 200
validation_samples = 16
max_words = 10000
embedding_dim = 16


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
    count_word_frequency = {}  # 存储文档中出现的词及其对应的出现次数

    for i in range(len(texts)):
        seg_content_data = jieba.cut(texts[i])
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in count_word_frequency:
                word_index_dic[word] = len(word_index_dic) + 1
            #     count_word_frequency[word] = 1
            # else:
            #     count_word_frequency[word] += 1
    print('统计文档中词的出现个数', word_index_dic)
    for j in range(len(texts)):
        word_list = []
        seg_content_data = jieba.cut(texts[j])
        for word in seg_content_data:
            seg_content_list[j].append(word)
            if word in word_index_dic.keys():
                word_list.append(word_index_dic[word])
            else:
                word_list.append(0)
        train_data.append(word_list)
    print(len(train_data))
    print(len(train_data[0]))
    print(train_data[0])

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)

    train_label = np.array(labels)
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    x_train = train_data[:training_samples]
    y_train = train_label[:training_samples]
    x_val = train_data[training_samples: training_samples + validation_samples]
    y_val = train_label[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val


def tokennize_test_data():
    labels, texts = process_data()
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts=texts)
    sequences = tokenizer.texts_to_sequences(texts=texts)
    x_test =  keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
    y_test = np.asarray(labels)

    return x_test, y_test


def tokennize_data():
    '''
    将text向量化，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    filename = 'train'
    labels, texts = process_data(filename)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts=texts)
    sequences = tokenizer.texts_to_sequences(texts=texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data =  keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, word_index


def parse_word_embedding(word_index):
    '''
    将预计算的词向量空间的word建立索引和矩阵
    :return:
    '''
    glove_dir = 'D:\\text2sequences\\glove.6B'

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 'r', encoding='UTF-8')
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


def train_model(x_train, y_train, x_val, y_val):
    '''
    训练模型
    :return:训练时loss,acc
    '''
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dense(1, activation=tf.nn.sigmoid))
    model.summary()

    # # 将GLOVE加载到模型中
    # x_train, y_train, x_val, y_val, word_index = tokennize_data()
    # # x_test, y_test = tokennize_test_data()
    # embedding_matrix = parse_word_embedding(word_index)
    # model.layers[0].set_weights([embedding_matrix])
    # model.layers[0].trainable = False

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])


    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=16,
                        validation_data=(x_val, y_val))
    # model.save('pre_trained_glove_model_1.h5')
    # plott_results()

    # 评估专用
    model.load_weights('pre_trained_glove_model_1.h5')
    test_acc = model.evaluate(x_test, y_test)
    print(test_acc)


def plott_results():
    '''
    作图
    '''
    history = train_model()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_val, y_val = peredata()
    train_model(x_train, y_train, x_val, y_val)
