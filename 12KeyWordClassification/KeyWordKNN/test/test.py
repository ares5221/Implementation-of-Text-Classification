#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #__file__的是打印当前被执行的模块.py文件相对路径，注意是相对路径
print(BASE_DIR)
sys.path.append(BASE_DIR)

from src.KNNmodel import test

if __name__ =='__main__':
    FEATURE = ['攻击行为', '违纪行为', '社会退缩', '学习问题']
    s_F = [1,2,3, 4]
    sentence = '孩子打架'
    test(s_F[0], sentence)
