#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re
from docx import Document

def change_file_name(documentsPath):
    '''
    输入待处理文件夹路径part3,根据文档的title重命名文件名
    将该文件夹下文件名称统一格式 a0000268 编织美丽故事.docx
    :return:
    '''

    n = 268
    for fname in os.listdir(documentsPath):
        if fname[-5:] == '.docx':
            # print(fname, type(fname))

            oldname = documentsPath + fname
            addname = 'a0000' + str(n) + ' '
            flag = 0
            newname = ''
            document = Document(oldname)
            for paragraph in document.paragraphs:
                firstLine = paragraph.text
                flag += 1
                if flag > 0:
                    # print('##############', paragraph.text)
                    newname = paragraph.text
                    break

            newfname = addname + newname + '.docx'
            print(newfname)
            newname = documentsPath + newfname
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
            n += 1
    print('文件夹中文件转换完毕，文件总数 = ', n)


if __name__ == '__main__':
    documentsPath = os.path.abspath('./A/')
    print('待修改文件名的文档路径：', documentsPath)
    change_file_name(documentsPath)
