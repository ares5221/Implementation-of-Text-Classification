#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import pandas as pd


def process_data(select):
    '''
    通过参数选择不同的数据，便于调用二级标签的数据
    :return: texts,labels
    '''
    ann_filename = r'label_text.csv'
    df = pd.read_csv(ann_filename, encoding='utf-8')  # pandas读取csv数据
    print('step1: 读取标注文本及标签信息', select)  # 显示当前处理的二级标签类别
    make_label(df)
    X = df[['Content']]
    Y = df.multilabel
    # print(df.head())  # 查看前5行
    classes_num = 0
    if select == 'all':
        anntxt = X.values.tolist()
        label = Y.tolist()
        train_x_list = anntxt
        train_y_list = [x - 1 for x in label]
        print(train_y_list)
        print(train_x_list)
        classes_num = 95

    if select == 'attack':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [0, 1, 2]:
                content.append(anntxt[i])
                tag.append(label[i])
        train_x_list = content
        train_y_list = tag
        classes_num = 3

    if select == 'disorder':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [3, 4, 5, 6]:
                content.append(anntxt[i])
                tag.append(label[i] - 3)
        train_x_list = content
        train_y_list = tag
        classes_num = 4

    if select == 'pinxingwenti':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [7, 8, 9]:
                content.append(anntxt[i])
                tag.append(label[i] - 7)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    return train_x_list, train_y_list, classes_num


def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


if __name__ == '__main__':
    select = 'pinxingwenti'
    anntxt, label = process_data(select)
    print(anntxt, label)
