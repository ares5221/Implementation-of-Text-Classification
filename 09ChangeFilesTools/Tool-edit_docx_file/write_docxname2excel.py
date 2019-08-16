#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re
from docx import Document
import xlsxwriter


def read_file_name(documentsPath):
    '''
    输入待处理文件夹路径part3,根据文档的title重命名文件名
    将该文件夹下文件名称统一格式 a0000268 编织美丽故事.docx
    :return:
    '''
    n = 0
    indexs, titles = [], []
    for fname in os.listdir(documentsPath):
        if fname[-5:] == '.docx':
            print(fname, type(fname))
            indexs.append(fname[:-5].split(' ')[0])
            titles.append(fname[:-5].split(' ')[1])
    print(indexs)
    print(titles)
    return indexs, titles


def save_excel(indexs, titles):
    workbook = xlsxwriter.Workbook('part3_labels.xlsx')
    worksheet = workbook.add_worksheet()
    row = 1
    col = 0
    for k in range(len(titles)):
        worksheet.write(row, col, indexs[k])
        worksheet.write(row, col + 1, titles[k])
        row += 1

    workbook.close()


if __name__ == '__main__':
    documentsPath = os.path.abspath('./A/')
    print('待修改文件名的文档路径：', documentsPath)
    indexs, titles = read_file_name(documentsPath)
    save_excel(indexs, titles)
