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
        # train_x_list = anntxt
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] not in [30, 31, 32, 51, 52]:
                content.append(anntxt[i])
                if label[i] > 32 and label[i] < 51:
                    tag.append(label[i] - 3)
                elif label[i] > 52:
                    tag.append(label[i] - 3 - 2)
                else:
                    tag.append(label[i])
        train_x_list = content
        train_y_list = tag
        print('获取all ann 数据规模：', len(train_y_list), train_y_list)
        print(train_x_list)
        classes_num = 90

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

    if select == 'buliangshihao':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [10, 11]:
                content.append(anntxt[i])
                tag.append(label[i] - 10)
        train_x_list = content
        train_y_list = tag
        classes_num = 2
        print(train_y_list)
        print(train_x_list)

    if select == 'tuisuo':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [12, 13, 14]:
                content.append(anntxt[i])
                tag.append(label[i] - 12)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'yiyuwenti':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [15, 16, 17]:
                content.append(anntxt[i])
                tag.append(label[i] - 15)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'jiaolvwenti':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [18, 19, 20]:
                content.append(anntxt[i])
                tag.append(label[i] - 18)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'ziwozhongxin':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [25, 26, 27]:
                content.append(anntxt[i])
                tag.append(label[i] - 25)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'xuexiwenti':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [21, 22, 23, 24]:
                content.append(anntxt[i])
                tag.append(label[i] - 21)
        train_x_list = content
        train_y_list = tag
        classes_num = 4
        print(train_y_list)
        print(train_x_list)

    if select == 'jiduanshijian':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [28, 29]:
                content.append(anntxt[i])
                tag.append(label[i] - 28)
        train_x_list = content
        train_y_list = tag
        classes_num = 2
        print(train_y_list)
        print(train_x_list)

    if select == 'jiankangzhuangkuang':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [33, 34]:
                content.append(anntxt[i])
                tag.append(label[i] - 33)
        train_x_list = content
        train_y_list = tag
        classes_num = 2
        print(train_y_list)
        print(train_x_list)

    if select == 'suoshuqunti':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [35, 36, 37]:
                content.append(anntxt[i])
                tag.append(label[i] - 35)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'jiatingjiegou':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [38, 39, 40]:
                content.append(anntxt[i])
                tag.append(label[i] - 38)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'jiaoyangfangshi':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [41, 42, 43, 44]:
                content.append(anntxt[i])
                tag.append(label[i] - 41)
        train_x_list = content
        train_y_list = tag
        classes_num = 4
        print(train_y_list)
        print(train_x_list)

    if select == 'jiatingqifen':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [45, 46, 47, 48]:
                content.append(anntxt[i])
                tag.append(label[i] - 45)
        train_x_list = content
        train_y_list = tag
        classes_num = 4
        print(train_y_list)
        print(train_x_list)

    if select == 'chengyuanjiankangzhuangkuang':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [49, 50]:
                content.append(anntxt[i])
                tag.append(label[i] - 49)
        train_x_list = content
        train_y_list = tag
        classes_num = 2
        print(train_y_list)
        print(train_x_list)

    if select == 'chengyuanjingjizhuangkuang':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [53, 54]:
                content.append(anntxt[i])
                tag.append(label[i] - 53)
        train_x_list = content
        train_y_list = tag
        classes_num = 2
        print(train_y_list)
        print(train_x_list)

    if select == 'jiaoshilingdaofangshi':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [55, 56, 57]:
                content.append(anntxt[i])
                tag.append(label[i] - 55)
        train_x_list = content
        train_y_list = tag
        classes_num = 3
        print(train_y_list)
        print(train_x_list)

    if select == 'tongbanjiena':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [58, 59, 60, 61, 62]:
                content.append(anntxt[i])
                tag.append(label[i] - 58)
        train_x_list = content
        train_y_list = tag
        classes_num = 5
        print(train_y_list)
        print(train_x_list)

    if select == 'genbenyuanyin':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]:
                content.append(anntxt[i])
                tag.append(label[i] - 65)
        train_x_list = content
        train_y_list = tag
        classes_num = 16
        print(train_y_list)
        print(train_x_list)

    if select == 'yurenduice':
        anntxt = X.values.tolist()
        label = Y.tolist()
        label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]:
                content.append(anntxt[i])
                tag.append(label[i] - 81)
        train_x_list = content
        train_y_list = tag
        classes_num = 14
        print(train_y_list)
        print(train_x_list)

    return train_x_list, train_y_list, classes_num


def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


if __name__ == '__main__':
    select = 'pinxingwenti'
    anntxt, label, classes_num = process_data(select)
    print(anntxt, label, classes_num)
