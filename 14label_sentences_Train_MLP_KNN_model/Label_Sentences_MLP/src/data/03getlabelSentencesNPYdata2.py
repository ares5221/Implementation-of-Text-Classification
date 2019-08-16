#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv, os
from bert_serving.client import BertClient

'''
将标注数据-学习问题的句子通过bert生成每个句子的embedding向量
便于在调用的时候可以直接导入
生成的数据保存在learn_sentences.npy learn_labels.npy
'''

def getTestNPYData(data, name):
    X = [[] for i in range(len(data))]
    bc = BertClient()
    for index in range(len(data)):
        vector = bc.encode([data[index]])
        X[index] = vector.tolist()
        if index % 100 == 0:
            print(index, 'is finish')
    train_data = np.array(X)
    Xnpyname = name + '_' + str(len(data)) + '.npy'
    np.save(Xnpyname, train_data)
    print('%s 的数据通过bert embedding 为向量npy格式' % name)


def getLabelTestNPYData(data,name):
    '''生成分类的标签信息'''
    print(len(data))
    if len(data) == 4 and name == 'learn_labels':
        print(len(data[0]), len(data[1]), len(data[2]), len(data[3]))
        label_learn = [0 for i in range(len(data[0]))] +[1 for i in range(len(data[1]))] +[2 for i in range(len(data[2]))] + [3 for i in range(len(data[3]))]
        num = len(data[0]) + len(data[1]) + len(data[2])+ len(data[3])
        np.save(name + '_' + str(num) + '.npy', label_learn)
        print(label_learn)
        print('%s 的标签通过存储为向量npy格式' %name)

# Start Position--->>>>>>>>>
if __name__ == '__main__':
    csvpath = 'label_text_pro.csv'  # 共获得7809条数据

    learning_ability = []
    learning_method = []
    learning_attitude = []
    learning_attention = []

    if os.path.exists(csvpath):
        with open(csvpath, newline='', encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                # print(row[0].split(',')[0])
                if row[0].split(',')[0] == 'label25':
                    learning_ability.append(row[0].split(',')[1])
                if row[0].split(',')[0] == 'label26':
                    learning_method.append(row[0].split(',')[1])
                if row[0].split(',')[0] == 'label27':
                    learning_attitude.append(row[0].split(',')[1])
                if row[0].split(',')[0] == 'label28':
                    learning_attention.append(row[0].split(',')[1])

    print(len(learning_ability), learning_ability)
    print(len(learning_method), learning_method)
    print(len(learning_attitude), learning_attitude)
    print(len(learning_attention), learning_attention)

    forKNNcompile_learn_sentences = learning_ability[:-1] + learning_method[:-5] + learning_attitude[:-37] + learning_attention[:-7]
    forKNNcompile_learn_label = [learning_ability[:-1], learning_method[:-5], learning_attitude[:-37], learning_attention[:-7]]

    getTestNPYData(forKNNcompile_learn_sentences, 'learn_sentences')
    getLabelTestNPYData(forKNNcompile_learn_label, 'learn_labels')