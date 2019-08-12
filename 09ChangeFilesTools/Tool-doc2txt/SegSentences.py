#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re

mypath = os.path.abspath('./txtfiles')
workpath = os.path.abspath('./worktxtfiles/')
all_FileNum = 0


def Translate(path):
    '''
    将一个目录下所有txt文件中按句号断句。
    段之间加一个换行符号相隔两行，句子中的。后面加一个换行符号。
    :param path: 待处理的文件夹
    :return:
    '''
    global all_FileNum
    files = os.listdir(path)  # 该目录下所有文件的名字
    for f in files:
        if (f[0] == '~' or f[0] == '.'):
            continue
        filepath = path + '\\' + f
        # print(os.path.basename(filepath)) # 该目录下所有文件的名字
        if filepath[-4:] == '.txt':
            pattern_rule = re.compile(r'。')
            pattern_rule2 = re.compile(r'\n')
            with open(filepath, 'r', encoding='utf-8') as f:  # 设置文件对象
                str = f.read()
                str = re.sub(pattern_rule2, '\n\n', str)
                str = re.sub(pattern_rule, '。\n', str)  # str = str.replace('。', '。\n')
                # print(str)
            # f = open(filepath,encoding='utf-8')
            # lines = f.readlines()
            # for line in lines:
            #     print(line)
            save_name = os.path.basename(filepath)  # 将txt文件name作为保存txt文件的name
            full_path = workpath + save_name
            with open(full_path, 'a', encoding='utf-8') as ff:
                ff.write(str)
                # ff.write('\n')
            all_FileNum += 1


if __name__ == '__main__':
    Translate(mypath)
    print('文件夹中文件转换完毕，文件总数 = ', all_FileNum)
    # if not os.path.exists(filename):  # 如果不存在该文件，重建
    #     pass
