#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv, os
from bert_serving.client import BertClient

'''
通过csv数据构建用于trainLearnSentencesModel的MLP数据
生成的数据保存在X_learn_sentences.npy  Y_learn_sentences.npy
'''


def get_atecQuestAns(filename):
    # 导入csv中数据
    print(filename)
    saveatecdata = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            if len(i) > 2:
                allqa = [i[0], i[1], i[2]]
                saveatecdata.append(allqa)
    return saveatecdata


def getData(filename):
    if filename == 'all_attack_words.csv':
        datasize = 12880
    else:
        datasize = 20864
    X = [[] for i in range(datasize)]
    Y = [[] for i in range(datasize)]
    data = get_atecQuestAns(filename)
    # print(data)
    bc = BertClient()
    for index in range(datasize):
        tmp = data[index]
        v1 = bc.encode([tmp[0]])
        v2 = bc.encode([tmp[1]])
        qq1_vec = np.append(v1, v2)
        X[index] = qq1_vec.tolist()
        Y[index] = int(tmp[2])
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

    Xnpyname = 'X_train_' + filename[:-4] + '_' + str(datasize) + '.npy'
    Ynpyname = 'Y_train_' + filename[:-4] + '_' + str(datasize) + '.npy'
    np.save(Xnpyname, X_train)
    np.save(Ynpyname, Y_train)
    print('%s 的数据已经处理为npy格式' % filename)


# Start Position--->>>>>>>>>
if __name__ == '__main__':
    csv_file_attack = 'all_attack_words.csv'  # 10298
    getData(csv_file_attack)
    csv_file_learn = 'all_learn_words.csv'  # 15748
    getData(csv_file_learn)
