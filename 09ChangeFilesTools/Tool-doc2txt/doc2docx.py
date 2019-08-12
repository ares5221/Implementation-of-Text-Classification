#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from win32com import client as wc
import os
import re

mypath = os.path.abspath('./docxfile-part2')
all_FileNum = 0


def Translate(path):
    global all_FileNum
    '''将一个目录下所有doc文件转成docx'''
    # 该目录下所有文件的名字
    files = os.listdir(path)
    for f in files:
        if (f[0] == '~' or f[0] == '.'):
            continue
        new = path + '\\' + f
        # 除去后边的.doc后缀
        #         # tmp = new[:-4]
        # print(tmp)
        # # 改成txt格式
        # word = wc.Dispatch('Word.Application')
        # print(word)
        # doc = word.Documents.Open(new)
        # doc.SaveAs(tmp + '.docx', 16)
        # # doc.SaveAs(sss, 4)
        # doc.Close()
        # all_FileNum = all_FileNum + 1
        if new[-4:] == '.doc':
            # filename = re.sub("\D", "", f)
            # 除去后边的.doc后缀
            print(new)
            tmp = new[:-4]
            word = wc.Dispatch('Word.Application')
            doc = word.Documents.Open(new)
            doc.SaveAs(tmp + '.docx', 16)
            doc.Close()
            all_FileNum = all_FileNum + 1


if __name__ == '__main__':
    Translate(mypath)
    print('文件总数 = ', all_FileNum)
