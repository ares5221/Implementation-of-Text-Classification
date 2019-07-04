#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from bert_serving.client import BertClient
import csv
import numpy as np
from ProcessData import process_data


def manage(index, sentence):
    '''
    根据一级维度标签选择不同的分类模型
    :param index:
    :return:
    '''
    if index == 1:
        classification_tag = KNN1(index, sentence)
    elif index == 2:
        classification_tag = KNN2(index, sentence)
    elif index == 3:
        classification_tag = KNN3(index, sentence)
    elif index == 4:
        classification_tag = KNN4(index, sentence)
    else:
        print('invalid index')
        return
    return classification_tag


def peredata(index):
    '''
    将句子通过BERT embedding
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    train_data = []  # 用于存储训练文档信息
    texts, labels, classes_num = process_data(index)
    print(texts)
    print(labels)
    print('step1: get text and label')
    tmp_data = np.array(texts)
    tmp_label = np.array(labels)
    indices = np.arange(len(labels))  # shuffle
    np.random.shuffle(indices)
    tmp_data = tmp_data[indices]
    tmp_label = tmp_label[indices]

    texts = tmp_data.tolist()
    labels = tmp_label.tolist()
    print(texts)
    print(labels)
    print('step2: shuffle data')

    # if not os.path.exists(select + '_train.csv'):
    #     with open(select + '_train.csv', 'a', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         for i in range(cut_num):
    #             data = [labels[i], texts[i][0]]
    #             writer.writerow(data)
    # if not os.path.exists(select + '_test.csv'):
    #     with open(select + '_test.csv', 'a', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         for i in range(cut_num, len(labels)):
    #             data = [labels[i], texts[i][0]]
    #             writer.writerow(data)

    if len(labels) < 2:
        print('样本数据规模不足')
        return
    else:
        bc = BertClient()
        for i in range(len(texts)):
            # print(texts[i], type(texts[i]))
            v1 = bc.encode([texts[i]])
            train_data.append(v1[0])
        train_label = np.array(labels)

        if not os.path.exists(str(index) + "_train_vec.npy"):  # save data
            np.save(str(index) + "_train_vec.npy", train_data)
        if not os.path.exists(str(index) + "_train_lab.npy"):  # save data
            np.save(str(index) + "_train_lab.npy", train_label)
        print('step2: 划分数据集后，BERT转换向量完成，')
    return train_data, train_label


def GetTrainData(index):
    train_data = np.load(str(index) + "_train_vec.npy")
    train_label = np.load(str(index) + "_train_lab.npy")
    print(index, '直接导入分类数据成功')
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    return train_data, train_label


def train_knnmodel(train_data, train_label, sentence):
    test_cal_lab = []  # 用于存储knn计算得到的label结果
    topk = 1  # knn中可以调节设置参数K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    for i in range(len(test_vec)):
        score = np.sum(test_vec[i] * train_data, axis=1) / np.linalg.norm(train_data, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        # print('当前待比较分类label-->content:', test_list[i])
        # for idx in topk_idx:
        #     # print('> %s\t%s' % (score[idx], idx), )
        #     print('###找到的相似-->', train_list[idx])
        #     with open('test&simsent.csv', 'a', newline='', encoding='utf-8') as csvfile:
        #         writer = csv.writer(csvfile)
        #         data = [test_list[i], train_list[idx]]
        #         writer.writerow(data)

        # topk_idx 存储最相似的数据id，通过该id获取对应的label，
        # 比较这些id得到其中数量最多的相同label作为测试数据的label
        cal_lab = []
        for idx in topk_idx:
            cal_lab.append(train_label[idx])
        print(cal_lab)
        # print(max(cal_lab, key=cal_lab.count))
        test_cal_lab.append(max(cal_lab, key=cal_lab.count))
    print(test_cal_lab)
    return test_cal_lab
    # print(test_lab[:len(test_lab)])


def KNN1(index, sentence):
    if not os.path.exists("1_train_vec.npy"):
        train_data, train_label = peredata(index)
    else:
        train_data, train_label = GetTrainData(index)  # 获取训练数据
    res = train_knnmodel(train_data, train_label, sentence)
    if res[0] == 0:
        return '身体攻击行为'
    if res[0] == 1:
        return '言语攻击行为'
    if res[0] == 2:
        return '关系攻击行为'


def KNN2(index, sentence):
    if not os.path.exists("2_train_vec.npy"):
        train_data, train_label = peredata(index)
    else:
        train_data, train_label = GetTrainData(index)  # 获取训练数据
    res = train_knnmodel(train_data, train_label, sentence)
    if res[0] == 0:
        return '隐蔽性违反课堂纪律行为'
    if res[0] == 1:
        return '扰乱课堂秩序行为'
    if res[0] == 2:
        return '违反课外纪律行为'


def KNN3(index, sentence):
    if not os.path.exists("3_train_vec.npy"):
        train_data, train_label = peredata(index)
    else:
        train_data, train_label = GetTrainData(index)  # 获取训练数据
    res = train_knnmodel(train_data, train_label, sentence)
    if res[0] == 0:
        return '言语型退缩'
    if res[0] == 1:
        return '行为型退缩'
    if res[0] == 2:
        return '心理型退缩'


def KNN4(index, sentence):
    if not os.path.exists("4_train_vec.npy"):
        train_data, train_label = peredata(index)
    else:
        train_data, train_label = GetTrainData(index)  # 获取训练数据
    res = train_knnmodel(train_data, train_label, sentence)
    if res[0] == 0:
        return '学习能力问题'
    if res[0] == 1:
        return '学习方法问题'
    if res[0] == 2:
        return '学习态度问题'
    if res[0] == 3:
        return '注意力问题'


def test(index, sentence):
    # print('sss')
    # print(index, sentence)
    cls_tag = manage(index, sentence)
    print(cls_tag)


if __name__ == '__main__':
    FEATURE = ['攻击行为', '违纪行为', '社会退缩', '学习问题']
    s_F = [1, 2, 3, 4]
    # sentence = '孩子打同学的脸'
    # sentence = '孩子打人'
    # sentence = '孩子骂人'
    # test(s_F[0], sentence)

    # sentence = '孩子睡觉'
    # sentence = '孩子上课打打闹闹'
    # sentence = '孩子随意讲话'
    # test(s_F[1], sentence)

    # sentence = '孩子沉默寡言'
    # sentence = '孩子不爱交朋友'
    # sentence = '孩子自卑'
    # test(s_F[2], sentence)

    sentence = '孩子学习不好'
    sentence = '孩子学习没有计划'
    sentence = '孩子不写作业'
    sentence = '孩子注意力不集中'
    test(s_F[3], sentence)
