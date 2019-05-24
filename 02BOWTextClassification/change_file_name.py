#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re

all_FileNum = 0


def Translate(workpath, index):
    '''
    将index.txt文件中按句号断句。段之间加一个换行符号相隔两行，句子中的。后面加一个换行符号。
    并将文件名称修改为id + title
    :param path: 待处理的文件name
    :return:
    '''
    global all_FileNum
    if index < 10:
        index = '000' + str(index)
    elif index >= 10 and index < 100:
        index = '00' + str(index)
    elif index >= 100 and index < 1000:
        index = '0' + str(index)
    else:
        index = str(index)
    filename = index + '.txt'
    print(filename)
    filepath = os.path.join(workpath, filename)
    print(filepath)
    # print(os.path.split(filename)[1])
    # # print(os.path.basename(filepath)) # 该目录下所有文件的名字
    # txtname = os.path.split(workpath)[1] +os.path.split(filename)[1]
    # txtname = os.path.join(workpath,txtname)
    # print(txtname)
    if filepath[-4:] == '.txt':
        pattern_rule = re.compile(r'。')
        pattern_rule2 = re.compile(r'\n')
        with open(filepath, 'r', encoding='utf-8') as f:  # 设置文件对象
            first_line = f.readline()  # 读取文本第一行作为其title
            text_name = first_line.strip()
            str1 = f.read()  # 读取全部文本按句号分段
            str1 = re.sub(pattern_rule2, '\n\n', str1)
            str1 = re.sub(pattern_rule, '。\n', str1)  # str = str.replace('。', '。\n')

        print('ss', os.path.basename(filepath), os.path.basename(filepath)[:-4])
        newname = os.path.basename(filepath)[:-4] + ' ' + text_name + '.txt'
        workpath = os.path.join(workpath, 'new/')  # 保存到该路径下new文件中
        print(workpath)
        full_path = workpath + newname
        # print(full_path)
        if not os.path.exists(full_path):
            with open(full_path, 'a', encoding='utf-8') as ff:
                ff.write(str1)
            all_FileNum += 1
        else:
            print('已经创建该文件，跳过')


if __name__ == '__main__':
    db_dir = r'G:\tf-start\Implementation-of-Text-Classification\dataset'
    filename = r'work/bzrlt/'  # 需要分类处理的文档路径
    # filename = r'work/dyal/'  # 需要分类处理的文档路径
    work_dir = os.path.join(db_dir, filename)
    Translate(work_dir, 20)
    print('文件夹中文件转换完毕，文件总数 = ', all_FileNum)
