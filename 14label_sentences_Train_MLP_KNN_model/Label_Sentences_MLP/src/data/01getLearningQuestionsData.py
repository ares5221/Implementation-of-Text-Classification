#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# 从label_text_pro数据中选取出学习问题相关的数据

import csv
import os

csvpath = 'label_text_pro.csv'  # 共获得7809条数据

learning_ability = []
learning_method = []
learning_attitude = []
learning_attention = []

if os.path.exists(csvpath):
    with open(csvpath, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            # print(row[0].split(',')[0])
            if row[0].split(',')[0] == 'label25':
                learning_ability.append(row[0].split(',')[1])
            if row[0].split(',')[0] == 'label26':
                learning_method.append(row[0].split(',')[1])
            if row[0].split(',')[0] == 'label27':
                learning_attitude.append(row[0].split(',')[1])
            if row[0].split(',')[0] == 'label28':
                learning_attention.append(row[0].split(',')[1])


print(len(learning_ability), learning_ability)
print(len(learning_method), learning_method)
print(len(learning_attitude), learning_attitude)
print(len(learning_attention), learning_attention)
all_learn_words = learning_ability + learning_method + learning_attitude + learning_attention
# 3.1将学习问题相关 标注语句 保存为csv文件
with open("all_learn_sentences.csv", "a", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    count_learn = 0
    for first in all_learn_words:
        for second in all_learn_words:
            if first == second:
                continue
            else:
                count_learn += 1
                if first in learning_ability and second in learning_ability:
                    label = 1
                elif first in learning_method and second in learning_method:
                    label = 1
                elif first in learning_attitude and second in learning_attitude:
                    label = 1
                elif first in learning_attention and second in learning_attention:
                    label = 1
                else:
                    label = 0
            # print(first, second, label)
            writer.writerow([first, second, label])
    print('---将学习问题 相关标注数据写入csv文件完成 共得到%d 条标记数据---' % count_learn) #190482
