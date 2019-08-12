#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import xlrd
from datetime import date,datetime

file = 'label标签.xlsx'


def read_excel():

    wb = xlrd.open_workbook(filename=file)#打开文件
    # print(wb.sheet_names())#获取所有表格名字
    # sheet1 = wb.sheet_by_index(0)#通过索引获取表格
    sheet1 = wb.sheet_by_name(wb.sheet_names()[0])#通过名字获取表格
    print(sheet1.name,sheet1.nrows,sheet1.ncols)
    old2new = {}
    for i in range(1, sheet1.nrows):
        for j in range(0,2):
            if sheet1.cell_value(i,1) not in old2new:
                old2new[sheet1.cell_value(i,1)] = sheet1.cell_value(i,0)
    print(old2new)

    # rows = sheet1.row_values(2)#获取行内容
    # cols = sheet1.col_values(3)#获取列内容
    # print(rows)
    # print(cols)
    #
    # print(sheet1.cell(1,0).value)#获取表格里的内容，三种方式
    # print(sheet1.cell_value(1,0))
    # print(sheet1.row(1)[0].value)

read_excel()