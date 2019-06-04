#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# 统计句子长度情况
def read_data(datapath):
    df = pd.read_csv(datapath, encoding='utf-8')  # pandas读取csv数据
    print(df.shape)  # 查看数据大小
    print(df.head())  # 查看前5行
    # print(df['Content'])
    X = df[['Content']]
    train_data = np.array(X)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    return train_x_list


def count(data):
    width = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    data_len = [0 for i in range(10)]
    for i in data:
        if len(i[0]) < width[0]:
            data_len[0] += 1
        if len(i[0]) > width[0] and len(i[0]) < width[1]:
            data_len[1] += 1
        if len(i[0]) > width[1] and len(i[0]) < width[2]:
            data_len[2] += 1
        if len(i[0]) > width[2] and len(i[0]) < width[3]:
            data_len[3] += 1
        if len(i[0]) > width[3] and len(i[0]) < width[4]:
            data_len[4] += 1
        if len(i[0]) > width[4] and len(i[0]) < width[5]:
            data_len[5] += 1
        if len(i[0]) > width[5] and len(i[0]) < width[6]:
            data_len[6] += 1
        if len(i[0]) > width[6] and len(i[0]) < width[7]:
            data_len[7] += 1
        if len(i[0]) > width[7] and len(i[0]) < width[8]:
            data_len[8] += 1
        if len(i[0]) > width[8] and len(i[0]) < width[9]:
            data_len[9] += 1
    print(data_len)
    return data_len


def draw(num_list):
    name_list = ['< 50', '50-100', '100-150', '150-200', '200-250', '250-300',
                 '300-350', '350-400', '400-450', '>450']
    rects = plt.bar(range(len(num_list)), num_list, color='rgby')
    # X轴标题
    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    index = [float(c) + 0.1 for c in index]
    plt.ylim(ymax=1117, ymin=0)
    plt.xticks(index, name_list)
    plt.ylabel("sentence num")  # X轴标签
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
    plt.show()


# START
if __name__ == '__main__':
    sourcepath = os.path.abspath('../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    ss = read_data(dataPath)
    length = count(ss)
    draw(length)
