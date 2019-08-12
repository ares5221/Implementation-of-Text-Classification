#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import os
import re

'''读取标注文件.ann,将其中label 对应语句信息保存在data.txt中'''


def read_ann(annFilePath):
    '''
        读取文件夹下ann文本的内容，读取其中title, 对应label
        :return: titles, labels
        '''
    # 新的第二部分标注数据109
    # annFilePath = os.path.abspath('../dataset/AnnFileForLabel/part2_ann_label')
    # # 旧的第一部分标注数据176
    # annFilePath = os.path.abspath('../dataset/AnnFileForLabel/part1_ann_label')
    print('标记数据文档路径：', annFilePath)
    datas = []
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.ann':
            ff = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            label_list = ff.readlines()
            if len(label_list) > 0:
                for item in label_list:
                    p = re.compile(r'label\d{1,2}')  # 正则表达式提取label
                    label_str_list = item.split('\t')  # label_str:T11	label32 238 239	他
                    if len(label_str_list) <= 1:
                        print(fname, '该标注文件有错误，需要修改一下空行的问题')
                    label = p.findall(label_str_list[1])[0]
                    val = label_str_list[2].replace('\n', '')
                    datas.append([label, val])
    print(datas)
    return datas


def save_csv(datas):
    for data in datas:
        with open('label_text.csv', 'a', newline='', encoding='utf-8') as csvfile:
            # 获得 writer对象 delimiter是分隔符 默认为 ","
            writer = csv.writer(csvfile)
            # 调用 writer的 writerow方法将 test_writer_data写入 test_writer.csv文件
            writer.writerow(data)


if __name__ == "__main__":
    part1_Path = os.path.abspath('../dataset/AnnFileForLabel/part1_ann_label')
    part2_Path = os.path.abspath('../dataset/AnnFileForLabel/part2_ann_label')
    datas1 = read_ann(part1_Path)
    datas2 = read_ann(part2_Path)
    count = len(datas1) + len(datas2)
    # save_csv(datas1)
    # save_csv(datas2)
    print('保存csv完毕 Finished！得到数据条数--->', count)
