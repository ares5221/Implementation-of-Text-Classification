#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv


def read_data():
    filename = 'clean_source_data.csv'
    count = 0
    with open(filename, 'r', encoding='utf-8') as fcsv:
        csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
        for ann_txt in csv_reader:
            print(ann_txt)
            if ann_txt[0] in ['label11', 'label12', 'label13']:  # gongji
                with open("./../data/1.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label11':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label12':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label13':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label14', 'label15', 'label16']:  # weiji
                with open("./../data/2.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label14':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label15':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label16':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label17', 'label18', 'label19']:  # buliang
                with open("./../data/3.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label17':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label18':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label19':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label20', 'label21', 'label22']:  # shehuituisuo
                with open("./../data/4.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label20':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label21':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label22':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label23', 'label24']:  # qingxu
                with open("./../data/5.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label23':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label24':
                        writer.writerow(['1', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label25', 'label26', 'label27', 'label28']:  # xuexiwenti
                with open("./../data/6.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label25':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label26':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label27':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label28':
                        writer.writerow(['3', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label29', 'label30', 'label31']:  # ziwozhongxin
                with open("./../data/7.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label29':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label30':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label31':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label32', 'label33', 'label34']:  # teshuwenti
                with open("./../data/8.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label32':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label33':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label34':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label4', 'label5', 'label6']:  # jiankangzhuangkuang
                with open("./../data/9.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label4':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label5':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label6':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label7', 'label8', 'label9', 'label10']:  # suoshuqunti
                with open("./../data/10.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label7':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label8':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label9':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label10':
                        writer.writerow(['3', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label35', 'label36', 'label37', 'label38']:  # jiatinhjiegou
                with open("./../data/11.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label35':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label36':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label37':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label38':
                        writer.writerow(['3', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label39', 'label40', 'label41', 'label42']:  # jiaoyangfangshi
                with open("./../data/12.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label39':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label40':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label41':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label42':
                        writer.writerow(['3', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label43', 'label44', 'label45', 'label46']:  # jiatingqifen
                with open("./../data/13.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label43':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label44':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label45':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label46':
                        writer.writerow(['3', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label47', 'label48', 'label49']:  # 成员健康
                with open("./../data/14.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label47':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label48':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label49':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label50', 'label51']:  # 成员wenhua
                with open("./../data/15.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label50':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label51':
                        writer.writerow(['1', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label52', 'label53']:  # 成员jingji
                with open("./../data/16.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label52':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label53':
                        writer.writerow(['1', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label55', 'label56', 'label57']:  # lingdaofangshi
                with open("./../data/17.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label55':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label56':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label57':
                        writer.writerow(['2', ann_txt[1]])
                    else:
                        pass
            if ann_txt[0] in ['label58', 'label59', 'label60', 'label61', 'label62']:  # tongbanjiena
                with open("./../data/18.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label58':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label59':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label60':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label61':
                        writer.writerow(['3', ann_txt[1]])
                    elif ann_txt[0] == 'label62':
                        writer.writerow(['4', ann_txt[1]])
                    else:
                        pass

            if ann_txt[0] in ['label66', 'label67', 'label68', 'label69', 'label70', 'label71', 'label72', 'label73',
                              'label74', 'label75', 'label76', 'label77']:  # xuqiuqueshi
                with open("./../data/19.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] == 'label66':
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] == 'label67':
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] == 'label68':
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] == 'label69':
                        writer.writerow(['3', ann_txt[1]])
                    elif ann_txt[0] == 'label70':
                        writer.writerow(['4', ann_txt[1]])
                    elif ann_txt[0] == 'label71':
                        writer.writerow(['5', ann_txt[1]])
                    elif ann_txt[0] == 'label72':
                        writer.writerow(['6', ann_txt[1]])
                    elif ann_txt[0] == 'label73':
                        writer.writerow(['7', ann_txt[1]])
                    elif ann_txt[0] == 'label74':
                        writer.writerow(['8', ann_txt[1]])
                    elif ann_txt[0] == 'label75':
                        writer.writerow(['9', ann_txt[1]])
                    elif ann_txt[0] == 'label76':
                        writer.writerow(['10', ann_txt[1]])
                    elif ann_txt[0] == 'label77':
                        writer.writerow(['11', ann_txt[1]])
                    else:
                        pass

            if ann_txt[0] in ['label78', 'label89', 'label80', 'label81',
                              'label82', 'label83', 'label84', 'label85',
                              'label86', 'label87', 'label88', 'label89',
                              'label90', 'label91', 'label92', 'label93',
                              'label94', 'label95', 'label96', 'label97',
                              'label98', 'label99', 'label100', 'label101',
                              'label102', 'label103', 'label104', 'label105',
                              'label106', 'label107', 'label108', 'label109',
                              'label110', 'label111', 'label112', 'label113', 'label114'
                              ]:  # duice
                with open("./../data/20.csv", "a", encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if ann_txt[0] in ['label78', 'label79', 'label80']:
                        writer.writerow(['0', ann_txt[1]])
                    elif ann_txt[0] in ['label81', 'label82', 'label83']:
                        writer.writerow(['1', ann_txt[1]])
                    elif ann_txt[0] in ['label84', 'label85', 'label86', 'label87']:
                        writer.writerow(['2', ann_txt[1]])
                    elif ann_txt[0] in ['label88', 'label89', 'label90']:
                        writer.writerow(['3', ann_txt[1]])
                    elif ann_txt[0] in ['label91', 'label92', 'label93', 'label94']:
                        writer.writerow(['4', ann_txt[1]])
                    elif ann_txt[0] in ['label95', 'label96', 'label97']:
                        writer.writerow(['5', ann_txt[1]])
                    elif ann_txt[0] in ['label98', 'label99']:
                        writer.writerow(['6', ann_txt[1]])
                    elif ann_txt[0] in ['label100', 'label101', 'label102', 'label103']:
                        writer.writerow(['7', ann_txt[1]])
                    elif ann_txt[0] in ['label104', 'label105', 'label106']:
                        writer.writerow(['8', ann_txt[1]])
                    elif ann_txt[0] == 'label107':
                        writer.writerow(['9', ann_txt[1]])
                    elif ann_txt[0] == 'label108':
                        writer.writerow(['10', ann_txt[1]])
                    elif ann_txt[0] in ['label109', 'label110']:
                        writer.writerow(['11', ann_txt[1]])
                    elif ann_txt[0] in ['label111', 'label112', 'label113']:
                        writer.writerow(['12', ann_txt[1]])
                    elif ann_txt[0] == 'label114':
                        writer.writerow(['13', ann_txt[1]])
                    else:
                        pass


if __name__ == '__main__':
    read_data()
