#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from ProcessData import process_data
from bert_serving.client import BertClient
import csv
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

'''
获取标注数据
将标注数据随机分成训练数据与测试数据8：2
全部转换为bert encoding向量
通过KNN比较测试数据属于那些标签，计算准确度，可控制k的大小
'''


def peredata(select):
    '''
    将句子通过BERT embedding转为vector，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    train_data = []  # 用于存储训练文档信息
    texts, labels, classes_num = process_data(select)
    print(texts)
    print(labels)
    tmp_data = np.array(texts)
    tmp_label = np.array(labels)
    indices = np.arange(len(labels))  # shuffle
    np.random.shuffle(indices)
    tmp_data = tmp_data[indices]
    tmp_label = tmp_label[indices]
    print(tmp_data[0])
    print(tmp_label[0])

    texts = tmp_data.tolist()
    labels = tmp_label.tolist()
    print(texts)
    print(labels)
    print('step2: 划分训练集与测试集数据')
    cut_num = round(len(labels) * 0.8)

    if not os.path.exists(select + '_train.csv'):
        with open(select + '_train.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(cut_num):
                data = [labels[i], texts[i][0]]
                writer.writerow(data)
    if not os.path.exists(select + '_test.csv'):
        with open(select + '_test.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(cut_num, len(labels)):
                data = [labels[i], texts[i][0]]
                writer.writerow(data)

    if cut_num < 6:
        print('样本数据规模不足')
        return
    else:
        bc = BertClient()
        for i in range(len(texts)):
            v1 = bc.encode(texts[i])
            train_data.append(v1[0])
        train_label = np.array(labels)

        train_vec = train_data[:cut_num]
        test_vec = train_data[cut_num:]
        train_lab = train_label[:cut_num]
        test_lab = train_label[cut_num:]
        print('切分数据集成功')

        if not os.path.exists(select + "_train_vec.npy"):  # save data
            np.save(select + "_train_vec.npy", train_vec)
        if not os.path.exists(select + "_test_vec.npy"):
            np.save(select + "_test_vec.npy", test_vec)
        if not os.path.exists(select + "_train_lab.npy"):  # save data
            np.save(select + "_train_lab.npy", train_lab)
        if not os.path.exists(select + "_test_lab.npy"):
            np.save(select + "_test_lab.npy", test_lab)
        print('step2: 划分数据集后，BERT转换向量完成，')
    return train_vec, test_vec, train_lab, test_lab, texts[:cut_num], texts[cut_num:]


def GetTrainData(select):
    train_vec = np.load(select + "_train_vec.npy")
    test_vec = np.load(select + "_test_vec.npy")
    train_lab = np.load(select + "_train_lab.npy")
    test_lab = np.load(select + "_test_lab.npy")
    print(select, '直接导入分类数据成功')
    print('Shape of data tensor:', train_vec.shape)
    print('Shape of label tensor:', test_vec.shape)
    # indices = np.arange(train_data.shape[0])  # shuffle
    # np.random.shuffle(indices)
    # train_data = train_data[indices]
    # train_label = train_label[indices]
    f = open(select + '_train.csv', 'r', encoding='utf-8')
    csvreader = csv.reader(f)
    train_list = list(csvreader)
    f2 = open(select + '_test.csv', 'r', encoding='utf-8')
    csvreader2 = csv.reader(f2)
    test_list = list(csvreader2)
    return train_vec, test_vec, train_lab, test_lab, train_list, test_list


def train_knnmodel(train_vec, test_vec, train_lab, test_lab, select, train_list, test_list):
    test_cal_lab = []  # 用于存储knn计算得到的label结果
    topk = 3   # knn中可以调节设置参数K setting

    for i in range(len(test_vec)):
        score = np.sum(test_vec[i] * train_vec, axis=1) / np.linalg.norm(train_vec, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        print('当前待比较分类label-->content:', test_list[i])
        for idx in topk_idx:
            # print('> %s\t%s' % (score[idx], idx), )
            print('###找到的相似-->', train_list[idx])
            with open('test&simsent.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                data = [test_list[i], train_list[idx]]
                writer.writerow(data)

        # topk_idx 存储最相似的数据id，通过该id获取对应的label，
        # 比较这些id得到其中数量最多的相同label作为测试数据的label
        cal_lab = []
        for idx in topk_idx:
            cal_lab.append(train_lab[idx])
        # print(cal_lab)
        # print(max(cal_lab, key=cal_lab.count))
        test_cal_lab.append(max(cal_lab, key=cal_lab.count))
    print(test_cal_lab)
    print(test_lab[:len(test_lab)])

    #计算精度
    right_num = 0
    for i in range(len(test_cal_lab)):
        if test_cal_lab[i] == test_lab[i]:
            right_num +=1
    acc = right_num/len(test_cal_lab)
    print('精度acc：', acc)


if __name__ == '__main__':
    print('采用BERT+KNN模型对标注文本按一级标签做分类')
    option = ['all']
    for i in range(0, len(option)):
        select = option[i]
        if not os.path.exists(select + "_train_vec.npy"):
            train_vec, test_vec, train_lab, test_lab, train_list, test_list = peredata(select)
        else:
            train_vec, test_vec, train_lab, test_lab, train_list, test_list = GetTrainData(select)  # 获取训练数据
        train_knnmodel(train_vec, test_vec, train_lab, test_lab, select, train_list, test_list)
