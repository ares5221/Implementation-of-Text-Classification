#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml.ns import qn
import re
'''将一个目录下所有docx,文件名用txt中的name'''
txtpath = os.path.abspath('./txtfolderFordocx')
docxpath = os.path.abspath('./docxfolder/')

all_FileNum = 0
index = 367
def ChangeDocxName(path):
    global all_FileNum
    files = os.listdir(path)  # 该目录下所有文件的名字
    for f in files:
        if (f[0] == '~' or f[0] == '.'):
            continue
        filepath = path + '\\' + f
        if filepath[-5:] == '.docx':
            document = Document(filepath)  # 打开docx文件
            flag = 0
            newname = ''
            for paragraph in document.paragraphs:
                firstLine = paragraph.text
                if firstLine.startswith('发布人') or '情况介绍' in firstLine or '案例描述' in firstLine:
                    flag -= 1
                flag += 1
                if flag >0:
                    # print('##############', paragraph.text)
                    newname = paragraph.text
                    if len(paragraph.text) > 20:
                        newname = paragraph.text[:20]
                    break
            newname = str(index + all_FileNum) + ' ' + newname+ '.docx'
            print(f, '@@@@@@@@@', newname)
            # os.rename(filepath, newname)
            all_FileNum += 1



if __name__ == '__main__':
    ChangeDocxName(txtpath)
    print('文件夹中文件转换完毕，文件总数 = ', all_FileNum)