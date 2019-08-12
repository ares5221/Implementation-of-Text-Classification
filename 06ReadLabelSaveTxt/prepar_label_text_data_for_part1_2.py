#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import os
import re

'''在新的标注方法下：读取标注文件.ann,将其中label 对应语句信息保存在data.txt中'''


def read_ann(annFilePath):
    '''
        读取文件夹下ann文本的内容，读取其中title, 对应label
        :return: titles, labels
        '''
    print('标记数据文档路径：', annFilePath)
    label_dict = {'label1': 'label11', 'label2': 'label12', 'label3': 'label13', 'label4': 'label14',
                  'label5': 'label16',
                  'label6': 'label15', 'label7': 'label16', 'label8': 'label17', 'label9': 'label18',
                  'label10': 'label19',
                  'label12': 'label32', 'label13': 'label20', 'label14': 'label21', 'label15': 'label22',
                  'label16': 'label23', 'label17': 'label23', 'label18': 'label23', 'label19': 'label24',
                  'label20': 'label24',
                  'label21': 'label24', 'label22': 'label25', 'label23': 'label26', 'label24': 'label27',
                  'label25': 'label28',
                  'label26': 'label29', 'label27': 'label30', 'label28': 'label31', 'label29': 'label33',
                  'label30': 'label34',
                  'label31': 'label1', 'label32': 'label2', 'label33': 'label3', 'label34': 'label5',
                  'label35': 'label6',
                  'label36': 'label8', 'label37': 'label9', 'label38': 'label10', 'label39': 'label35',
                  'label40': 'label36',
                  'label41': 'label37', 'label42': 'label39', 'label43': 'label40', 'label44': 'label41',
                  'label45': 'label42',
                  'label46': 'label43', 'label47': 'label44', 'label48': 'label45', 'label49': 'label46',
                  'label50': 'label47',
                  'label51': 'label48', 'label52': 'label54', 'label53': 'label51', 'label54': 'label52',
                  'label55': 'label53',
                  'label56': 'label55', 'label57': 'label56', 'label58': 'label57', 'label59': 'label58',
                  'label60': 'label59',
                  'label61': 'label60', 'label62': 'label61', 'label63': 'label62', 'label64': 'label63',
                  'label65': 'label64',
                  'label66': 'label66', 'label67': 'label76', 'label68': 'label67', 'label69': 'label68',
                  'label70': 'label69',
                  'label71': 'label70', 'label72': 'label71', 'label73': 'label73', 'label74': 'label73',
                  'label75': 'label72',
                  'label76': 'label74', 'label77': 'label74', 'label78': 'label75', 'label79': 'label76',
                  'label80': 'label76',
                  'label81': 'label77', 'label82': 'label78', 'label83': 'label79', 'label84': 'label80',
                  'label85': 'label81',
                  'label86': 'label82', 'label87': 'label83', 'label88': 'label84', 'label89': 'label85',
                  'label90': 'label86',
                  'label91': 'label87', 'label92': 'label88', 'label93': 'label89', 'label94': 'label90',
                  'label95': 'label91',
                  'label96': 'label92', 'label97': 'label93', 'label98': 'label94', 'label99': 'label95',
                  'label100': 'label96',
                  'label101': 'label97', 'label102': 'label98', 'label103': 'label99', 'label104': 'label100',
                  'label105': 'label101', 'label106': 'label102', 'label107': 'label103', 'label108': 'label104',
                  'label109': 'label105', 'label110': 'label106', 'label111': 'label107', 'label112': 'label108',
                  'label113': 'label109', 'label114': 'label110', 'label115': 'label111', 'label116': 'label112',
                  'label117': 'label113', 'label118': 'label114'}

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
                    if label in label_dict:
                        label = label_dict[label]
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
    part1_Path = os.path.abspath('../dataset/AnnFileForLabel/part1_ann_label')
    part2_Path = os.path.abspath('../dataset/AnnFileForLabel/part2_ann_label')
    datas1 = read_ann(part1_Path)
    datas2 = read_ann(part2_Path)
    count = len(datas1) + len(datas2)
    save_csv(datas1)
    save_csv(datas2)
    print('保存csv完毕 Finished！得到数据条数--->', count)
