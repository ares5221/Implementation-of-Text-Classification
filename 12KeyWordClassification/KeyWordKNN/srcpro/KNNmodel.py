#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from bert_serving.client import BertClient
import numpy as np
from ProcessData import process_data
from sklearn.neighbors import KNeighborsClassifier

def manage(sentence):
    if len(sentence) >0:
        classification_tag = KNN(sentence)
    else:
        print('invalid sentence')
        return
    return classification_tag


def peredata():
    '''
    将句子通过BERT embedding
    :return:train_data, train_label
    '''
    train_data = []
    texts, labels, classes_num = process_data()
    # print(texts)
    # print(labels)
    print('step1: get text and label')
    tmp_data = np.array(texts)
    tmp_label = np.array(labels)
    indices = np.arange(len(labels))  # shuffle
    np.random.shuffle(indices)
    tmp_data = tmp_data[indices]
    tmp_label = tmp_label[indices]

    texts = tmp_data.tolist()
    labels = tmp_label.tolist()
    print('after shuffle data')
    print(texts)
    print(labels)

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
        if not os.path.exists("key_train_vec.npy"):  # save data
            np.save("key_train_vec.npy", train_data)
        if not os.path.exists("key_train_lab.npy"):  # save data
            np.save("key_train_lab.npy", train_label)
        print('step2: 划分数据集后，BERT转换向量完成，')
    return train_data, train_label


def GetTrainData():
    train_data = np.load("key_train_vec.npy")
    train_label = np.load("key_train_lab.npy")
    # print(index, '直接导入分类数据成功')
    # print('Shape of data tensor:', train_data.shape)
    # print('Shape of label tensor:', train_label.shape)
    return train_data, train_label


def train_knnmodel(train_data, train_label, sentence):
    topk = 3  # K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    for i in range(len(test_vec)):
        score = np.sum(test_vec[i] * train_data, axis=1) / np.linalg.norm(train_data, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        cal_lab = []
        for idx in topk_idx:
            cal_lab.append(train_label[idx])
        sent_predict = max(cal_lab, key=cal_lab.count)
    return sent_predict


def train_KNNmodel(train_data, train_label, sentence):
    topk = 3  # K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    for i in range(len(test_vec)):
        knn_classifier = KNeighborsClassifier(topk)
        knn_classifier.fit(train_data, train_label)
        y_predict = knn_classifier.predict(test_vec)
    return y_predict


def KNN(sentence):
    if not os.path.exists("key_train_vec.npy"):
        train_data, train_label = peredata()
    else:
        train_data, train_label = GetTrainData()
    # res = train_knnmodel(train_data, train_label, sentence)
    res = train_KNNmodel(train_data, train_label, sentence)
    if res == 0:
        return ['gong_ji_xing_wei','身体攻击行为']
    if res == 1:
        return ['gong_ji_xing_wei','言语攻击行为']
    if res == 2:
        return ['gong_ji_xing_wei','关系攻击行为']
    if res == 3:
        return ['wei_ji_xing_wei','隐蔽性违反课堂纪律行为']
    if res == 4:
        return ['wei_ji_xing_wei','扰乱课堂秩序行为']
    if res == 5:
        return ['wei_ji_xing_wei','违反课外纪律行为']
    if res == 6:
        return ['she_hui_tui_suo','言语型退缩']
    if res == 7:
        return ['she_hui_tui_suo','行为型退缩']
    if res == 8:
        return ['she_hui_tui_suo','心理型退缩']
    if res == 9:
        return ['xue_xi_wen_ti','学习能力问题']
    if res == 10:
        return ['xue_xi_wen_ti','学习方法问题']
    if res == 11:
        return ['xue_xi_wen_ti','学习态度问题']
    if res == 12:
        return ['xue_xi_wen_ti','注意力问题']



def test(sentence):
    cls_tag = manage(sentence)
    print(cls_tag)


# sample to use
if __name__ == '__main__':
    for sentence in ['孩子打同学的脸', '孩子打人', '孩子骂人']:
        test(sentence)

    for sentence in ['孩子睡觉', '孩子上课打打闹闹', '孩子随意讲话']:
        test(sentence)

    for sentence in ['孩子沉默寡言', '孩子不爱交朋友', '孩子自卑']:
        test(sentence)

    for sentence in ['孩子学习不好', '孩子学习没有计划', '孩子不写作业', '孩子注意力不集中']:
        test(sentence)


'''
['gong_ji_xing_wei', '身体攻击行为']
['gong_ji_xing_wei', '身体攻击行为']
['gong_ji_xing_wei', '言语攻击行为']
['wei_ji_xing_wei', '隐蔽性违反课堂纪律行为']
['wei_ji_xing_wei', '扰乱课堂秩序行为']
['wei_ji_xing_wei', '隐蔽性违反课堂纪律行为']
['she_hui_tui_suo', '言语型退缩']
['wei_ji_xing_wei', '隐蔽性违反课堂纪律行为']
['she_hui_tui_suo', '心理型退缩']
['wei_ji_xing_wei', '隐蔽性违反课堂纪律行为']
['xue_xi_wen_ti', '学习态度问题']
['wei_ji_xing_wei', '隐蔽性违反课堂纪律行为']
['xue_xi_wen_ti', '注意力问题']
'''
