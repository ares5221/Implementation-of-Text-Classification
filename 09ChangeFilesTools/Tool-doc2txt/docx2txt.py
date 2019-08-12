#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from docx import Document
import re

'''将一个目录下所有docx文件转成txt,文件名用docx中的数字序号，生成一个对应的空的ann文件'''
path1 = os.path.abspath('./docxfile-part1')
path2 = os.path.abspath('./docxfile-part2')
path3 = os.path.abspath('./docxfile-part3')
all_FileNum = 0


def Translate(path):
    global all_FileNum
    files = os.listdir(path)  # 该目录下所有文件的名字
    for f in files:
        if (f[0] == '~' or f[0] == '.'):
            continue
        filepath = path + '\\' + f
        if filepath[-5:] == '.docx':
            print(f)
            filename = re.sub("\D", "", f)
            # filename = int(filename) + 174  # part1中包含174篇文档part2从175开始
            # filename = str(filename)
            document = Document(filepath)  # 打开docx文件
            for paragraph in document.paragraphs:
                # print(paragraph.text)     # 打印各段落内容文本
                savename = filename + '.txt'  # 将docx文件序号作为保存txt文件的文件名
                with open(savename, 'a', encoding='utf-8') as ff:
                    ff.write(paragraph.text)
                    ff.write('\n')
                annname = filename + '.ann'  # 生成对应的空.ann文件
                with open(annname, 'a', encoding='utf-8') as ff:
                    ff.write('')
            with open(savename, 'a', encoding='utf-8') as fff:
                fff.write('健康  一般儿童  完整家庭  平静型家庭气氛  和谐型家庭气氛  父母健康  一般型')

            all_FileNum += 1


if __name__ == '__main__':
    # Translate(path1)
    # Translate(path2)
    Translate(path3)
    print('文件夹中文件转换完毕，文件总数 = ', all_FileNum)
