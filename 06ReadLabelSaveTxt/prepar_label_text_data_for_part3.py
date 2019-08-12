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
    print('标记数据文档路径：', annFilePath)
    datas = []
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.ann':
            ff = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            label_list = ff.readlines()
            if len(label_list) > 0:
                for item in label_list:
                    p = re.compile(r'label[A-Za-z0-9]+')  # 正则表达式提取label
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
        with open('label_text_pro.csv', 'a', newline='', encoding='utf-8') as csvfile:
            # 获得 writer对象 delimiter是分隔符 默认为 ","
            writer = csv.writer(csvfile)
            # 调用 writer的 writerow方法将 test_writer_data写入 test_writer.csv文件
            writer.writerow(data)


if __name__ == "__main__":
    part3_Path = os.path.abspath('../dataset/AnnFileForLabel/part3_ann_label')
    datas3 = read_ann(part3_Path)
    count = len(datas3)
    save_csv(datas3)
    print('保存csv完毕 Finished！得到数据条数--->', count)
