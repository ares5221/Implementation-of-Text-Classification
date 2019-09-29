#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import pkuseg, jieba
import json


# BERT将一个句子分词后，每个词转为768向量后，输入LSTM+MLP做分类

def peredata(index):
    '''
    将句子分词后根据自己的词表转为[1,3,4,0,0,6,777,0]的表示
    通过BERT embedding转为vector
    :return:
    '''
    ann_filename = './data_LSTM/' + str(index) + '.csv'
    contents, tags = [], []
    with open(ann_filename, 'r', encoding='utf-8') as fcsv:
        csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:
            contents.append(row[1])
            tags.append(int(row[0]))
    classes_num = max(tags) + 1  # 类别个数
    print('1 read data ok...', len(contents), contents)

    if not os.path.exists('./count_sentence/' + str(index) + "count_sentence.csv"):
        max_len = get_data_info(contents, index)
    else:
        max_len = 0
        with open('./count_sentence/' + str(index) + "count_sentence.csv", 'r', encoding='utf-8') as fcsv:
            csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
            for i, rows in enumerate(csv_reader):
                if i == 1:
                    max_len = rows[1]

    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    sentences_seg_list = [[] for index in range(len(contents))]
    qinghuaSeg = pkuseg.pkuseg()
    for i in range(len(contents)):
        seg_content_data = qinghuaSeg.cut(contents[i])
        sentences_seg_list[i] = seg_content_data
        # 将出现的词编码，并将词与id的对应关系保存在word_index_dic
        for word in seg_content_data:
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic) + 1
    if not os.path.exists('./count_sentence/' + str(index) + 'word_index_dict.json'):
        with open('./count_sentence/' + str(index) + 'word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)

    max_words = len(word_index_dic)
    sentences_data = []
    for ssl in range(len(sentences_seg_list)):
        word_vec_list = []
        for word in sentences_seg_list[ssl]:
            word_vec_list.append(word_index_dic[word])
        sentences_data.append(word_vec_list)
    # shuffle
    train_data = np.array(sentences_data)
    train_label = np.array(tags)
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]
    # save data in npy
    np.save('./data_LSTM/' + str(index) + "_train_data.npy", train_data)
    np.save('./data_LSTM/' + str(index) + "_train_label.npy", train_label)
    return train_data, train_label, classes_num, max_words, max_len


def get_data_info(data, index):
    '''
    用于统计待处理的句子信息，包括句子的最大分词后的词的个数，所有句子分词后词的平均个数
    :param data:
    :return:统计得到句子分词后的信息后存储在对应文件种
    '''
    max_sentences_words_num = 0
    sentences_words_count = 0  # 统计全部句子的分词长度，计算句子分词后的平均词的个数
    qinghuaSeg = pkuseg.pkuseg()
    for i in range(len(data)):
        seg_content_data = qinghuaSeg.cut(data[i])
        sentences_words_count += len(seg_content_data)
        if len(seg_content_data) > max_sentences_words_num:
            max_sentences_words_num = len(seg_content_data)
        if i % 10 == 0:
            print('已经完成进度', i / len(data))
    with open('./count_sentence/' + str(index) + "count_sentence.csv", "a", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([index, max_sentences_words_num, sentences_words_count / len(data)])
    print('统计得到句子分词后的最大词的个数：', max_sentences_words_num, '平均词的个数：', sentences_words_count / len(data))
    return max_sentences_words_num


def GetTrainData(index):
    train_data = np.load('./data_LSTM/' + str(index) + "_train_data.npy")
    train_label = np.load('./data_LSTM/' + str(index) + "_train_label.npy")
    print(index, '直接导入分类数据成功')
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    # shuffle
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]
    # 统计分类类别
    set_label = set(train_label)
    classes_num = len(set_label)
    print('数据共分为几类的情况：', set_label, classes_num)

    count = [0 for i in range(classes_num)]
    for label in train_label:
        for count_index in range(classes_num):
            if label == count_index:
                count[count_index] += 1
    print('第 ', index, ' 个维度共有 ', len(train_label), '条数据，其中类别比例为 ', count)

    with open('./count_sentence/' + str(index) + 'word_index_dict.json', 'r', encoding='utf-8') as f:
        word_index_dic = json.load(f)
    max_len = 0
    with open('./count_sentence/' + str(index) + "count_sentence.csv", 'r', encoding='utf-8') as fcsv:
        csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
        for i, rows in enumerate(csv_reader):
            if i == 0:
                max_len = int(rows[1])
    return train_data, train_label, classes_num, len(word_index_dic), max_len


def train_model_keras(train_data, train_label, classes_num, index, max_word_num, max_len):
    if index == 20:
        classes_num = 14
    print('start build MLP model with keras...')
    epochs_num = 100
    embed_size = 768  # 词向量维度
    batch_size_num = 64
    max_len = max_len
    max_words = max_word_num + 1
    # print(train_data[0])
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)
    # print(train_data[0])
    train_label = np_utils.to_categorical(train_label, classes_num)

    embedding_matrix = np.zeros((max_words, embed_size))
    with open('./data_LSTM/' + str(index) + 'index_bert_dict.json', 'r', encoding='utf-8') as f:
        index_to_bert_vector = json.load(f)
        for key in index_to_bert_vector:
            index = int(key)
            if index == max_words:
                embedding_matrix[0] = np.array(index_to_bert_vector.get(key))
            else:
                embedding_matrix[index] = np.array(index_to_bert_vector.get(key))
    # bulid model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, embed_size))
    model.add(keras.layers.LSTM(128, activation=tf.nn.sigmoid, return_sequences=True))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(classes_num, activation=tf.nn.softmax))
    model.summary()
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_test_split_num = int(len(train_label) * 0.8)
    train_val_split_num = int(train_test_split_num * 0.8)
    x_val = train_data[train_val_split_num:train_test_split_num]
    partial_x_train = train_data[0:train_val_split_num]
    y_val = train_label[train_val_split_num:train_test_split_num]
    partial_y_train = train_label[0:train_val_split_num]
    test_data = train_data[train_test_split_num:]
    test_labels = train_label[train_test_split_num:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs_num,
                        batch_size=batch_size_num,
                        validation_data=(x_val, y_val),
                        shuffle=True)

    results = model.evaluate(test_data, test_labels)
    print(index, ' -->keras model评估模型效果(损失-精度）：...', results)
    print('step6: predict test data for count...')
    predictions = model.predict(test_data, batch_size=batch_size_num)
    predict = np.argmax(predictions, axis=1)
    print(predictions)
    print('预测结果如下：')
    print(predict)
    print('实际结果如下：')
    print(test_labels)
    return results


def get_embedding_dict(index):
    bc = BertClient()
    index_bert_vec_dic = {}  # 存储全部文档中出现的词及对应的bert embedding vector
    with open('./count_sentence/' + str(index) + 'word_index_dict.json', 'r', encoding='utf-8') as f:
        word_index_dic = json.load(f)
        # 将出现的词转为bert embedding vector
        for key, values in word_index_dic.items():
            if values not in index_bert_vec_dic and key is not ' ':
                index_bert_vec_dic[values] = bc.encode([key])[0].tolist()
    if not os.path.exists('./data_LSTM/' + str(index) + "index_bert_dict.json"):
        with open('./data_LSTM/' + str(index) + 'index_bert_dict.json', 'w', encoding='utf-8') as f:
            json.dump(index_bert_vec_dic, f, ensure_ascii=False)
    print('step3：分词结束得到%s个词，分词结果及编码保存在word_bert_dict.npy' % (len(word_index_dic)))


if __name__ == '__main__':
    print('采用BERT+MLP模型对标注文本做分类...')
    option = [i for i in range(1, 21)]  # 数据文件名索引为1-20
    for index in option:
        if index == 17 or index == 15:
            continue
        if True:
            # if index==15:
            if os.path.exists('./data_LSTM/' + str(index) + "_train_data.npy") and os.path.exists(
                    './data_LSTM/' + str(index) + "_train_label.npy"):
                train_data, train_label, classes_num, max_words, max_len = GetTrainData(index)  # 获取训练数据
            else:
                train_data, train_label, classes_num, max_words, max_len = peredata(index)

            if not os.path.exists('./data_LSTM/' + str(index) + "index_bert_dict.json"):
                get_embedding_dict(index)
            keras_sum = 0
            for i in range(1):
                # train in keras-Impl
                keras_acc = train_model_keras(train_data, train_label, classes_num, index, max_words, max_len)
                keras_sum += keras_acc[1]
            with open('./result/' + "keras_acc.csv", "a", encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([index, keras_sum / 1.0])
