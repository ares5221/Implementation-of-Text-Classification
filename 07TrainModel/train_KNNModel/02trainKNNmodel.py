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
    print('step2: 划分训练集与测试集数据')
    cut_num = round(len(labels) * 0.8)

    if not os.path.exists('train.csv'):
        with open('train.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(cut_num):
                data = [labels[i], texts[i][0]]
                writer.writerow(data)
    if not os.path.exists('test.csv'):
        with open('test.csv', 'a', newline='', encoding='utf-8') as csvfile:
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
        print('切分数据集：')
        train_vec = train_data[:cut_num]
        test_vec = train_data[cut_num:]
        train_lab = train_label[:cut_num]
        test_lab = train_label[cut_num:]

        # indices = np.arange(len(train_data))  # shuffle
        # np.random.shuffle(indices)
        # train_data = train_data[indices]
        # train_label = train_label[indices]
        if not os.path.exists(select + "_train_vec.npy"):  # save data
            np.save(select + "_train_vec.npy", train_vec)
        if not os.path.exists(select + "_test_vec.npy"):
            np.save(select + "_test_vec.npy", test_vec)
        if not os.path.exists(select + "_train_lab.npy"):  # save data
            np.save(select + "_train_lab.npy", train_lab)
        if not os.path.exists(select + "_test_lab.npy"):
            np.save(select + "_test_lab.npy", test_lab)
        print('step2: 划分数据集后，BERT转换向量完成，')
    return train_vec, test_vec, train_lab, test_lab


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
    return train_vec, test_vec, train_lab, test_lab


def train_knnmodel(train_vec, test_vec, train_lab, test_lab, select):
    print('aa')


if __name__ == '__main__':
    print('采用BERT+KNN模型对标注文本做分类')
    # option = ['all', 'attack', 'disorder', 'pinxingwenti', 'buliangshihao', 'tuisuo','yiyuwenti', 'jiaolvwenti', 'ziwozhongxin', 'xuexiwenti', 'jiduanshijian', 'jiankangzhuangkuang', 'suoshuqunti', 'jiatingjiegou', 'jiaoyangfangshi', 'jiatingqifen', 'chengyuanjiankangzhuangkuang', 'chengyuanjingjizhuangkuang', 'tongbanjiena', 'genbenyuanyin', 'yurenduice']
    option = ['all']
    for i in range(0, len(option)):
        select = option[i]
        if not os.path.exists(select + "_train_vec.npy"):
            train_vec, test_vec, train_lab, test_lab = peredata(select)
        else:
            train_vec, test_vec, train_lab, test_lab = GetTrainData(select)  # 获取训练数据
        train_knnmodel(train_vec, test_vec, train_lab, test_lab, select)