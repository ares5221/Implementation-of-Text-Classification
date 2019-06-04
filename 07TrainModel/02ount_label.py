#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import os
import pandas as pd


# 统计数据中label出现次数的情况
def read_data(datapath):
    df = pd.read_csv(datapath, encoding='utf-8')  # pandas读取csv数据
    print(df.shape)  # 查看数据大小
    # print(df.head())#查看前5行
    # print(df['Label'])
    # print(df['Content'])
    make_label(df)
    # print(df.head())
    # print(df['multilabel'])
    X = df[['Content']]
    Y = df.multilabel
    # print(X,Y)
    train_data = np.array(X)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    train_label = np.array(Y)  # np.ndarray()
    train_y_list = train_label.tolist()  # list
    # print(train_x_list)
    # print(train_y_list)

    return train_x_list, train_y_list


def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


def count_label(data):
    print(data)
    count = [0 for i in range(0, 95)]
    for i in data:
        print(i)
        count[i - 1] += 1
    print(count)


# START
if __name__ == '__main__':
    sourcepath = os.path.abspath('../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    print('ss', dataPath)
    content, train_label = read_data(dataPath)
    count_label(train_label)
