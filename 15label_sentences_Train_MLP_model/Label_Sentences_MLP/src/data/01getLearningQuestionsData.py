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
#共得到学习问题相关标注语句437条，留出50条作为最终测试系统的数据

trainMLP_learn_sentences = learning_ability[:-1] + learning_method[:-5] + learning_attitude[:-37] + learning_attention[:-7]
trainMLP_learn_labels = [0 for i in range(len(learning_ability[:-1]))] +\
                        [1 for i in range(len(learning_method[:-5]))] +\
                        [2 for i in range(len(learning_attitude[:-37]))] +\
                        [3 for i in range(len(learning_attention[:-7]))]

testMLP_learn_sentences = learning_ability[-1:] + learning_method[-5:] + learning_attitude[-37:] + learning_attention[-7:]
testMLP_learn_labels =[0 for i in range(1)] + [1 for i in range(5)]+[2 for i in range(37)]+[3 for i in range(7)]

# 将学习问题相关 标注语句 标签为0-3，生成训练MLP模型的数据，保存为csv文件
with open("trainMLP_learn_sentences.csv", "a", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    count_learn = 0
    for index in range(len(trainMLP_learn_sentences)):
        test_sentence = trainMLP_learn_sentences[index]
        test_label = trainMLP_learn_labels[index]
        print(index, test_sentence, test_label)
        writer.writerow([test_sentence, test_label])
    print('---Train将学习问题 相关标注数据写入csv文件完成 共得到%d 条标记数据---' % count_learn) #149342

# 原标注数据6/46/324/61 留出比例为1/5/37/7
with open("testMLP_learn_sentences_labels.csv", "a", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    count_learn = 0
    for index in range(len(testMLP_learn_sentences)):
        test_sentence = testMLP_learn_sentences[index]
        test_label = testMLP_learn_labels[index]
        print(index, test_sentence, test_label)
        writer.writerow([test_sentence, test_label])
    print('---Test将学习问题 相关标注数据写入csv文件完成 共得到%d 条标记数据---' % count_learn) #190482
