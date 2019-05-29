#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os


def change_similary_file_path(index1, index2):
    '''
    将找到的相似文档保存在另一个文件路径下
    index1, index2 两篇文档的索引
    :return:
    '''
    print('##############')
    documentsPath = os.path.abspath('../dataset/DocumentsForSimilarity/')
    print('待修改文件名的文档路径：', documentsPath)
    filename_list = os.listdir(documentsPath)
    newfile_dir = str(index1) + '&' + str(index2)+'\\'
    if not os.path.exists(documentsPath + newfile_dir):
        os.mkdir(documentsPath+ newfile_dir)

    # filename1 = documentsPath + filename_list[index1]
    # filename2 = documentsPath + filename_list[index2]
    # # print(filename1, filename2)
    # repath1 = documentsPath + newfile_dir + filename_list[index1]
    # repath2 = documentsPath + newfile_dir + filename_list[index2]
    # # print(repath1, repath2)
    # os.rename(filename1, repath1)
    # os.rename(filename2, repath2)
    print(filename_list[index1], '======>', filename_list[index2], '保存在：' + newfile_dir)


if __name__ == '__main__':
    change_similary_file_path(1, 2)
