#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os


def change_file_name():
    '''
    输入待处理文件夹路径
    将该文件夹下文件名称统一格式 不足四位的自动补0 如0007.txt 0052.txt 0152.txt 1152.txt
    :return:
    '''
    documentsPath = os.path.abspath('../dataset/DocumentsForFullName/')
    print('待修改文件名的文档路径：', documentsPath)
    filename_list = os.listdir(documentsPath)
    print(filename_list)
    # filename_list.sort(key=lambda x:(x[:-4]))
    # print(filename_list.sort())
    n = 0
    for fname in filename_list:
        if fname[-4:] == '.txt':
            print(fname[:-4], type(fname[:-4]))
            if int(fname[:-4]) < 10:
                oldname = documentsPath + filename_list[n]
                newname = documentsPath + '000' + fname[:-4] + '.txt'
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
            if int(fname[:-4]) >= 10 and int(fname[:-4]) < 100:
                oldname = documentsPath + filename_list[n]
                newname = documentsPath + '00' + fname[:-4] + '.txt'
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
            if int(fname[:-4]) >= 100 and int(fname[:-4]) < 1000:
                oldname = documentsPath + filename_list[n]
                newname = documentsPath + '0' + fname[:-4] + '.txt'
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
            if int(fname[:-4]) >= 1000:
                oldname = documentsPath + filename_list[n]
                newname = documentsPath + fname[:-4] + '.txt'
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1


if __name__ == '__main__':
    change_file_name()
