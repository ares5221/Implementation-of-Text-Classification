#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os


def change_file_name():
    '''
    输入待处理文件夹路径
    将该文件夹下文档统一修改name格式如：282.txt，序号自动递增
    :return:
    '''
    documentsPath = os.path.abspath('../dataset/DocumentsForChangeName/')
    print('待修改文件名的文档路径：', documentsPath)
    startIndex = 284 # 当前文档282篇，索引号从282开始
    filename_list = os.listdir(documentsPath)
    print(filename_list)
    # filename_list.sort(key=lambda x:(x[:-4]))
    # print(filename_list.sort())
    n = 0
    for fname in filename_list:
        if fname[-4:] == '.txt':
            print(fname)
            # oldname = documentsPath + filename_list[n]
            # # print(startIndex +n)
            # newname = documentsPath + str(startIndex +n) + '.txt'
            #
            # os.rename(oldname, newname)
            # print(oldname, '======>', newname)
            # n += 1



if __name__ == '__main__':
    change_file_name()