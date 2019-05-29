#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from text_similarity import cosSim
import os

#比较两个文件夹DocumentsForSimilarity，DocumentsForSimilarity2下文件，
# 将DocumentsForSimilarity文件夹中与DocumentsForSimilarity2中重复的文件转移到needdelete文件夹中
def process_data(doc_dir):
    '''
    读取文件夹下所有文本的内容，保存在texts数组中
    :return: texts
    '''
    documentsPath = os.path.abspath('../dataset/')
    # doc_dir = 'DocumentsForSimilarity'
    dir_name = os.path.join(documentsPath, doc_dir)
    print('待比较文档路径：', dir_name)
    texts = []
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            print(fname)
            f = open(os.path.join(dir_name, fname), 'r', encoding='UTF-8')
            tmp = f.read()
            texts.append(tmp)
            f.close()
    print('读取到', len(texts), '篇文档')
    return texts


def find_similarity(data_old, data_new):
    data_old_num = len(data_old)
    data_new_num = len(data_new)
    text_similarity = cosSim()
    file_index = []
    for i in range(0, data_old_num):
        max_sim = 0
        max_idx = 0
        for j in range(0, data_new_num):
            # print(datas[i], type(datas[i]))
            txt1, txt2 = data_old[i], data_new[j]
            ss = text_similarity.CalcuSim([txt1, txt2])
            #找到最相似的一篇文档，将其id记住
            if ss > max_sim:
                max_sim = ss
                max_idx = j
                print('maybe相似的文档是：', i, '<----->', j, '<----->', ss)
        if max_sim > 0.95:  # 若相似度大于0.95，则输出文件名
            print('相似的文档是：', i, '<----->', max_idx, '<----->', max_sim)
            file_index.append(max_idx)
    with open('file_index.txt', 'w', encoding='utf-8') as f:
        f.write(str(file_index))
    similary_file_repath(file_index)


def similary_file_repath(list_idx):
    documentsPath = os.path.abspath('../dataset/DocumentsForSimilarity/')
    print('待修改文件名的文档路径：', documentsPath)
    filename_list = os.listdir(documentsPath)
    newfile_dir = 'needdelete''\\'
    if not os.path.exists(documentsPath + newfile_dir):
        os.mkdir(documentsPath + newfile_dir)
    for i in list_idx:
        filename = documentsPath + filename_list[i]
        repath = documentsPath + newfile_dir + filename_list[i]
        os.rename(filename, repath)
        print(filename, '======>', repath, '保存在：' + newfile_dir)


if __name__ == "__main__":
    old_dir = 'DocumentsForSimilarity2'
    new_dir = 'DocumentsForSimilarity'
    texts_old = process_data(old_dir)
    texts_new = process_data(new_dir)
    find_similarity(texts_old, texts_new)
    print('所有文档的相似度已经比较完成！')
    #  #test
    # a = cosSim()
    # r = a.CalcuSim(["你好奥众，今天是星期三", "你好奥迪，今天是星期五"])
    # print(r)
