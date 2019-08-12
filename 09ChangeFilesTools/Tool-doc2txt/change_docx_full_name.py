#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re


def change_file_name(documentsPath):
    '''
    输入待处理文件夹路径
    将该文件夹下文件名称统一格式 a0000001 编织美丽故事.docx
    :return:
    '''

    isPart1 = os.path.split(os.path.split(documentsPath)[0])[1] == 'docxfile-part1'  #判断当前文件夹是否是part1，part1中文件名格式用.、而part2用空格
    n = 0
    for fname in os.listdir(documentsPath):
        if fname[-5:] == '.docx':
            # print(fname, type(fname))
            if isPart1:
                pattern = re.compile(r'^\d{1,3}[.、]')  # 匹配正式标题前的数字(最多三位)及.或者、
                index = pattern.findall(fname)
                index = int(index[0][:-1])
            else:
                pattern = re.compile(r'^\d{1,3}[ ]')  # 匹配正式标题前的数字(最多三位)及 一个空格
                index = pattern.findall(fname)
                index = int(index[0][:-1])
            # print(index, type(index))
            if index < 10:
                oldname = documentsPath + fname
                addname = 'a000000' + str(index) + ' '
                newfname = re.sub(pattern, addname, fname)
                newname = documentsPath + newfname
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
            if index >= 10 and index < 100:
                oldname = documentsPath + fname
                addname = 'a00000' + str(index) + ' '
                newfname = re.sub(pattern, addname, fname)
                newname = documentsPath + newfname
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
            if index >= 100 and index < 1000:
                oldname = documentsPath + fname
                addname = 'a0000' + str(index) + ' '
                newfname = re.sub(pattern, addname, fname)
                newname = documentsPath + newfname
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
            if index >= 1000:
                oldname = documentsPath + fname
                addname = 'a000' + str(index) + ' '
                newfname = re.sub(pattern, addname, fname)
                newname = documentsPath + newfname
                os.rename(oldname, newname)
                print(oldname, '======>', newname)
                n += 1
    print('文件夹中文件转换完毕，文件总数 = ', n)


if __name__ == '__main__':
    # documentsPath = os.path.abspath('./docxfile-part1/')
    documentsPath = os.path.abspath('./docxfile-part2/')
    print('待修改文件名的文档路径：', documentsPath)
    change_file_name(documentsPath)
