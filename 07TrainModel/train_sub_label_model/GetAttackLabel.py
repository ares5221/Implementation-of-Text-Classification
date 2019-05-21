#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# 获取攻击相关标签的数据
import pandas as pd
import numpy as np
import os
import csv


def read_data(datapath):
    df = pd.read_csv(datapath, encoding='utf-8')  # pandas读取csv数据
    print(df.shape)  # 查看数据大小
    print(df.head())  # 查看前5行
    # print(df['Label'])
    # print(df['Content'])
    make_label(df)
    print(df.head())
    # print(df['multilabel'])
    X = df[['Content']]
    Y = df.multilabel
    train_data = np.array(X)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    train_label = np.array(Y)  # np.ndarray()
    train_y_list = train_label.tolist()  # list
    print(train_x_list)
    print(train_y_list)
    return train_x_list, train_y_list


def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


def get_attack_label(content, train_label):
    attack = []
    attack_label = []
    for i in range(len(train_label)):
        if train_label[i] > 0 and train_label[i] < 4:
            attack.append(content[i][0])
            attack_label.append(train_label[i])
    print(len(attack_label), attack_label)
    print(len(attack), attack)
    return attack, attack_label


def save_attack_csv(attack_content, attack_label):
    # 输出数据写入CSV文件
    with open('attack_data.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['attack_label', 'attack_content'])
        for i in range(len(attack_label)):
            print([attack_label[i], attack_content[i]])
            csv_writer.writerow([attack_label[i], attack_content[i]])


if __name__ == '__main__':
    sourcepath = os.path.abspath('../../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    print('ss', dataPath)
    all_content, all_label = read_data(dataPath)

    attack_content, attack_label = get_attack_label(all_content, all_label)

    save_attack_csv(attack_content, attack_label)
