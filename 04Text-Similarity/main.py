#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from text_similarity import cosSim
import os

'''比较docs路径下所有文本之间的相似度，若相似度大于0.8，则输出文件名'''


def process_data():
    '''
    读取文件夹下所有文本的内容，保存在texts数组中
    :return: texts
    '''
    documentsPath = os.path.abspath('../dataset/')
    doc_dir = 'DocumentsForSimilarity'
    dir_name = os.path.join(documentsPath, doc_dir)
    print('待比较文档路径：', dir_name)
    texts = []
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            print(fname)
            f = open(os.path.join(dir_name, fname), 'r', encoding='UTF-8')
            tmp = f.read()
            # print(len(tmp), type(tmp))
            # if len(tmp) > 200:
            #     print(fname)
            #     tmp = tmp[:200]
            # print('#############', tmp)
            texts.append(tmp)
            f.close()
    print('读取到', len(texts), '篇文档')
    return texts


def find_similarity(datas):
    data_num = len(datas)
    text_similarity = cosSim()
    for i in range(data_num):
        for j in range(data_num):
            if i == j:
                # print('自己和自己不比较啦，因为相似度是1')
                continue
            else:
                # print(datas[i], type(datas[i]))
                txt1, txt2 = datas[i], datas[j]
                # print(datas[i+1], type(datas[i+1]))
                ss = text_similarity.CalcuSim([txt1, txt2])
                if ss > 0.75:
                    print(ss)
                if ss > 0.9:
                    print('相似的文档是：', i, '<----->', j, '<----->', ss)


if __name__ == "__main__":
    texts = process_data()
    find_similarity(texts)
    print('所有文档的相似度已经比较完成！')
    #  #test
    # a = cosSim()
    # r = a.CalcuSim(["你好奥众，今天是星期三", "你好奥迪，今天是星期五"])
    # print(r)
