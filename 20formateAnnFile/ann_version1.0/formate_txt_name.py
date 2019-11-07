#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import csv
import os
import re


def read_ann1(annFilePath):
    # 修改part1 中txt文件名从0001.txt改为a0000001.txt
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.txt':
            if True:
                oldname = os.path.join(annFilePath, fname)
                newname = annFilePath + '/new/' + 'a000' + fname
                print(oldname)
                print(newname)
                os.rename(oldname, newname)


def read_ann2(annFilePath):
    start_index, count = 175, 1
    # 修改part2 中txt文件名从0001.txt改为a0000176.txt
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.txt':
            if True:
                oldname = os.path.join(annFilePath, fname)
                newname = annFilePath + '/new/' + 'a0000' + str(start_index + count) + '.txt'
                print(oldname)
                print(newname)
                os.rename(oldname, newname)
                count += 1


def read_ann3(annFilePath):
    start_index, count = 267, 1
    # 修改part2 中txt文件名从0001.txt改为a0000268.txt
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.txt':
            if True:
                oldname = os.path.join(annFilePath, fname)
                newname = annFilePath + '/new/' + 'a0000' + str(start_index + count) + '.txt'
                # print(oldname)
                # print(newname)
                os.rename(oldname, newname)
                count += 1


if __name__ == "__main__":
    part1_Path = os.path.abspath('./part1_ann_label')
    part2_Path = os.path.abspath('./part2_ann_label')
    part3_Path = os.path.abspath('./part3_ann_label')
    # read_ann1(part1_Path)
    # read_ann2(part2_Path)
    read_ann3(part3_Path)
    print('保存csv完毕 Finished！得到数据条数--->')
