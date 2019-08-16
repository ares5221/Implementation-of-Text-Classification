#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv, os
from bert_serving.client import BertClient

'''
通过trainMLP_learn_sentences.csv数据构建用于trainLearnSentencesModel的MLP数据
生成的数据保存在X_learn_sentences.npy  Y_learn_sentences.npy
拆分为训练集及测试集
'''


def get_label_sentences_data(filename):
    # 导入csv中数据
    print(filename)
    saveatecdata = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            allqa = [i[0], i[1]]
            saveatecdata.append(allqa)
    return saveatecdata


def getData(filename):
    if filename == 'trainMLP_learn_sentences.csv':
        datasize = 387
        split_num = 387
    else:
        datasize = 50
        split_num = 50
    X = [[] for i in range(datasize)]
    Y = [[] for i in range(datasize)]
    data = get_label_sentences_data(filename)
    # print(data)
    bc = BertClient()
    for index in range(datasize):
        tmp = data[index]
        vector = bc.encode([tmp[0]])
        X[index] = vector[0]
        Y[index] = int(tmp[1])
        if index % 100 == 0:
            print(index, 'is finish')
            # print(tmp[0], tmp[1], tmp[2])
    X_train = np.array(X)
    Y_train = np.array(Y)
    print(X_train.shape)
    print(Y_train.shape)
    # shuffle
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    # train & test
    train_sentences_data = X_train[0:split_num]
    train_label_data = Y_train[0:split_num]
    test_sentences_data = X_train[0:split_num]
    test_label_data = Y_train[0:split_num]

    # XtrainName = 'X_train_' + 'learn_sentences_' + str(datasize) + '.npy'
    # YtrainName = 'Y_train_' + 'learn_sentences_' + str(datasize) + '.npy'
    # np.save(XtrainName, train_sentences_data)
    # np.save(YtrainName, train_label_data)

    XtestName = 'X_test_' + 'learn_sentences_' + str(datasize) + '.npy'
    YtestName = 'Y_test_' + 'learn_sentences_' + str(datasize) + '.npy'
    np.save(XtestName, test_sentences_data)
    np.save(YtestName, test_label_data)
    print('%s 的数据已经处理为npy格式' % filename)


# Start Position--->>>>>>>>>
if __name__ == '__main__':
    # train_file_learn = 'trainMLP_learn_sentences.csv'  # 190482
    # getData(train_file_learn)
    test_file_learn = 'testMLP_learn_sentences_labels.csv'  # 190482
    getData(test_file_learn)