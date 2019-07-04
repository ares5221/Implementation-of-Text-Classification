#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import pandas as pd


def process_data(select):
    '''
    按照一级标签做分类
    :return: texts,labels
    '''
    ann_filename = r'label_text.csv'
    df = pd.read_csv(ann_filename, encoding='utf-8')  # pandas读取csv数据
    print('step1: 读取标注文本及标签信息', select)
    make_label(df)
    X = df[['Content']]
    Y = df.multilabel
    print(df.head())  # 查看前5行
    classes_num = 0
    if select == 'all':
        anntxt = X.values.tolist()
        label = Y.tolist()
        # label = [x - 1 for x in label]
        content, tag = [], []
        for i in range(len(label)):
            if label[i] in [1,2,3]:
                content.append(anntxt[i])
                tag.append(0)
            if label[i] in [4,5, 6,7]:
                content.append(anntxt[i])
                tag.append(1)
            if label[i] in [8,9, 10]:
                content.append(anntxt[i])
                tag.append(2)
            if label[i] in [13,14, 15]:
                content.append(anntxt[i])
                tag.append(3)
            if label[i] in [16,17, 18, 19, 20,21]: #抑郁 焦虑问题合并
                content.append(anntxt[i])
                tag.append(4)
            if label[i] in [22, 23, 24,25]:#学习问题
                content.append(anntxt[i])
                tag.append(5)
            if label[i] in [26, 27, 28]:
                content.append(anntxt[i])
                tag.append(6)
            if label[i] in [12, 29, 30]: #特殊问题
                content.append(anntxt[i])
                tag.append(7)
            if label[i] in [34, 35]: #健康状况
                content.append(anntxt[i])
                tag.append(8)
            if label[i] in [36, 37, 38]: #所属群体	一般儿童，留守儿童，流动儿童，孤困儿童
                content.append(anntxt[i])
                tag.append(9)
            if label[i] in [39, 40, 41]: #家庭结构	寄养家庭，重组家庭，单亲家庭，完整家庭
                content.append(anntxt[i])
                tag.append(10)
            if label[i] in [42, 43, 44, 45]: #教养方式	权威型教养方式，专制型教养方式，溺爱型教养方式，忽视型教养方式
                content.append(anntxt[i])
                tag.append(11)
            if label[i] in [46, 47, 48, 49]: #家庭气氛	平静型家庭气氛，和谐型家庭气氛，冲突型家庭气氛，离散型家庭气氛
                content.append(anntxt[i])
                tag.append(12)
            if label[i] in [50, 51]:  # 成员健康状况
                content.append(anntxt[i])
                tag.append(13)
            if label[i] in [54, 55]:  # 成员经济状况
                content.append(anntxt[i])
                tag.append(14)
            if label[i] in [56, 57, 58]:  # 教师领导方式	权威型领导方式，民主型领导方式，放任型领导方式
                content.append(anntxt[i])
                tag.append(15)
            if label[i] in [59, 60, 61, 62, 63]:  #同伴接纳	受欢迎，一般型，被拒绝，被忽视，矛盾型
                content.append(anntxt[i])
                tag.append(16)
            if label[i] in [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]:
                content.append(anntxt[i])
                tag.append(17)
            if label[i] in [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]:
                content.append(anntxt[i])
                tag.append(18)

        train_x_list = content
        train_y_list = tag
        print('获取all ann 数据规模：', len(train_y_list))
        classes_num = 19
    return train_x_list, train_y_list, classes_num


def make_label(df):
    '''把label22，转换为22；字符截取 转为int。'''
    df["multilabel"] = df["Label"].apply(lambda x: int(x[5:]))


if __name__ == '__main__':
    select = 'all'
    anntxt, label, classes_num = process_data(select)
    print('获取数据完成', select)
