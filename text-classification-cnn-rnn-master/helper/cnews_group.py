#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

import os

def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def save_file(dirname):
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    print(os.path.abspath(''))
    f_train = open(r'data/cnews/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('data/cnews/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('data/cnews/cnews.val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):   # 分类目录
        print(category)
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 5000:
                f_train.write(category + '\t' + content + '\n')
            elif count < 6000:
                f_test.write(category + '\t' + content + '\n')
            else:
                f_val.write(category + '\t' + content + '\n')
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == '__main__':
    path = os.path.abspath('../..')
    filePath = path + '/dataset/train'
    print(path)
    save_file(filePath)

    print(len(open('data/cnews/cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('data/cnews/cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('data/cnews/cnews.val.txt', 'r', encoding='utf-8').readlines()))
