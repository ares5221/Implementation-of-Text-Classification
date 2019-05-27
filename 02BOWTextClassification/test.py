#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json
import os
import jieba
import numpy as np
from change_file_name_save import Translate
import pickle

db_dir = r'G:\tf-start\Implementation-of-Text-Classification\dataset'
filename = r'work\bzrlt'  # 需要分类处理的文档路径
filename = r'work\dyal'  # 需要分类处理的文档路径
filename = r'work\wenku'  # 需要分类处理的文档路径
# filename = r'work\wenku_deyuanli'  # 需要分类处理的文档路径
work_dir = os.path.join(db_dir, filename)


def get_predict_data():
    texts = []
    for fname in os.listdir(work_dir):
        if fname[-4:] == '.txt':
            f = open(os.path.join(work_dir, fname), 'r', encoding='UTF-8')
            tmp = f.read()
            # if len(tmp) > 200:
            #     print(fname)
            #     tmp = tmp[0:200]
            # print('#############',len(tmp))
            texts.append(tmp)
            f.close()
    return texts


def pereworkdata(data):
    word_dic = []
    f_word_dic = open("word_dic.txt", "r", encoding='utf-8')  # 设置文件对象
    str_word = f_word_dic.read()  # 将txt文件的所有内容读入到字符串str_word中
    str_word = str_word[1:-1] # 字符串处理
    for sw in str_word.split(','):
        sw = sw.lstrip().lstrip('\'').rstrip('\'')
        word_dic.append(sw)
    f_word_dic.close()  # 将文件关闭
    print('读取到词表的大小为： ',len(word_dic))
    print(word_dic)
    text_num = len(data)
    print('文本个数为：', text_num)
    #对文本分词
    seg_content_list = [[] for index in range(text_num)]
    for i in range(text_num):
        seg_content_data = jieba.cut(data[i])
        for word in seg_content_data:
            seg_content_list[i].append(word)
    print(seg_content_list)
    #编码
    word_len = len(word_dic)
    work_data = [[] for i in range(text_num)]
    for j in range(text_num):
        word_list = [0 for x in range(word_len)]
        index = 0
        for k in range(len(seg_content_list[j])):
            if seg_content_list[j][k] in word_dic:
                index = word_dic.index(seg_content_list[j][k])
                word_list[index] = 1
            else:
                word_list[index] = 0
        work_data[j] = word_list
        # print(j, sum(work_data[j]),work_data[j])
    return work_data


if __name__ == '__main__':
    if 1 == 0:
        worktexts = get_predict_data()
        data = pereworkdata(worktexts)
        train_data = np.array(data)
    # if not os.path.exists('work1.npy'):
    #    np.save("work1.npy", train_data)
    # if not os.path.exists('work2.npy'):
    #    np.save("work2.npy", train_data)
    if not os.path.exists('work3.npy'):
       np.save("work3.npy", train_data)
    # if not os.path.exists('work4.npy'):
    #    np.save("work4.npy", train_data)
    train_data = np.load("work3.npy")
    print('Shape of data tensor:', train_data.shape)
    # 导入读取模型`
    with open('LRModel.pickle', 'rb') as f:
        LRMModel = pickle.load(f)
        # 测试读取后的Model
    print(LRMModel.predict(train_data))
    test_acc = LRMModel.predict(train_data)

    # with open('SVMModel.pickle', 'rb') as f:
    #     SVMModel = pickle.load(f)
    #     # 测试读取后的Model
    # print(SVMModel.predict(train_data))
    # test_acc = SVMModel.predict(train_data)
    for i in range(len(test_acc)):
        # print(test_acc[i])
        if test_acc[i] > 0.8:
            print('文件：', i, '...', test_acc[i])
            Translate(work_dir, i)

    print('over')
