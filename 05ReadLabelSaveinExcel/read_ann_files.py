#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re
import xlsxwriter  # write xlsx   # xlwt 只能写xls

'''读取标注文件.ann,将其中label信息保存在excel中'''


def read_ann():
    '''
    读取文件夹下ann文本的内容，读取其中title, 对应label
    :return: titles, labels
    '''
    # 新的第二部分标注数据109
    annFilePath = os.path.abspath('../dataset/AnnFileForLabel/part3_ann_label')
    # # 新的第二部分标注数据109
    # annFilePath = os.path.abspath('../dataset/AnnFileForLabel/part2_ann_label')
    # # 旧的第一部分标注数据176
    # annFilePath = os.path.abspath('../dataset/AnnFileForLabel/part1_ann_label')
    print('待比较文档路径：', annFilePath)
    titles = []
    labels = []
    doc_num = 0
    text_index = []  # 存储文章id
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.txt':
            f = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            title = f.readline()  # 读取第一行为title
            title = title.replace('\n', '').strip()
            print('id--->', fname, ' title--->', title)
            titles.append(title)
            doc_num += 1
            text_index.append(fname[:-4])
            f.close()
        if fname[-4:] == '.ann':
            ff = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            label_list = ff.readlines()
            print('###', label_list)
            label = convert_label2value(label_list)
            print('name--->', fname, 'label--->', label)
            ss = []  # 去除重复的标签
            for s_label in label:
                # print('####', s_label)
                new_label_val = list(set(s_label))
                # print(new_label_val)
                ss.append(new_label_val)
            labels.append(ss)
            ff.close()
    print('读取到', doc_num, '篇文档', len(titles), len(labels))
    return titles, labels, text_index


def convert_label2value(label_list):
    len_data = len(label_list)
    label_val = [[] for i in range(28)]
    label_val[14] = ['一般儿童']

    for i in range(len_data):
        label_str = label_list[i]
        if True:
            # print(i, label_str)
            p = re.compile(r'label\d{1,2}')  # 正则表达式提取label
            label_str_list = label_str.split('\t')  # label_str:T11	label32 238 239	他
            print('ss', label_str_list)
            # label = label_str_list[1].split()
            # print('ss', len(label))

            label = p.findall(label_str_list[1])[0]
            val = label_str_list[2].replace('\n', '')
            # print(label, type(label), val, type(val))

            # 处理基础信息格式
            if label == 'label31':  # 年级
                if '学生' in val:
                    val = val.replace('学生', '')
                label_val[11] = [val]
            if label == 'label32':  # 性别
                if '她' in val or '女' in val:
                    # print('this is girl')
                    val = '女'
                else:
                    # print('this is boy')
                    val = '男'
                label_val[10] = [val]
            if label == 'label33':  # 年龄
                if '年龄' in val:
                    val = val.replace('年龄', '')
                    if '：' in val:
                        val = val.replace('：', '')
                        if '，' in val:
                            val = val.replace('，', '')
                label_val[12] = [val]

            if label == 'label64':  # 大众传媒影响
                label_val[24] = [val]
            if label == 'label65':  # 社会文化社会风气
                label_val[25] = [val]

            if label == 'label1':  # 攻击行为/
                label_val[0].append('身体攻击行为')
            if label == 'label2':
                label_val[0].append('言语攻击')
            if label == 'label3':
                label_val[0].append('间接攻击')

            if label == 'label4':  # 违纪行为/
                label_val[1].append('隐蔽性违纪行为')
            if label == 'label5':
                label_val[1].append('违反课堂制度行为')
            if label == 'label6':
                label_val[1].append('扰乱课堂秩序行为')
            if label == 'label7':
                label_val[1].append('违反校规校级行为')

            if label == 'label8':  # 品行问题/
                label_val[2].append('欺骗行为')
            if label == 'label9':
                label_val[2].append('偷盗行为')
            if label == 'label10':
                label_val[2].append('背德行为')

            if label == 'label11':  # 不良嗜好/
                label_val[3].append('强迫行为')
            if label == 'label12':
                label_val[3].append('沉迷行为')

            if label == 'label13':  # 退缩/
                label_val[4].append('言语性退缩')
            if label == 'label14':
                label_val[4].append('行为性退缩')
            if label == 'label15':
                label_val[4].append('心理性退缩')

            if label == 'label16':  # 抑郁问题/
                label_val[5].append('抑郁情绪问题')
            if label == 'label17':
                label_val[5].append('抑郁行为问题')
            if label == 'label18':
                label_val[5].append('抑郁思维问题')

            if label == 'label19':  # 焦虑问题/
                label_val[6].append('焦虑情绪问题')
            if label == 'label20':
                label_val[6].append('焦虑行为问题')
            if label == 'label21':
                label_val[6].append('焦虑思维问题')

            if label == 'label26':  # 自我中心/
                label_val[7].append('自我吹嘘型问题')
            if label == 'label27':
                label_val[7].append('执拗型问题')
            if label == 'label28':
                label_val[7].append('自私型问题')

            if label == 'label22':  # 学习问题/
                label_val[8].append('学习能力问题')
            if label == 'label23':
                label_val[8].append('学习方法问题')
            if label == 'label24':
                label_val[8].append('学习态度问题')
            if label == 'label25':
                label_val[8].append('注意力问题')

            if label == 'label29':  # 极端事件/
                label_val[9].append('青春期问题')
            if label == 'label30':
                label_val[9].append('极端问题')

            if label == 'label34':  # 健康状况/
                label_val[13].append('生理疾病')
            if label == 'label35':
                label_val[13].append('心理疾病')

            if label == 'label36':  # 所属群体/默认 一般
                label_val[14] = ['留守儿童']
            if label == 'label37':
                label_val[14] = ['流动儿童']
            if label == 'label38':
                label_val[14] = ['孤困儿童']

            if label == 'label39':  # 家庭结构/
                label_val[15].append('寄养家庭')
            if label == 'label40':
                label_val[15].append('重组家庭')
            if label == 'label41':
                label_val[15].append('单亲家庭')

            if label == 'label42':  # 教养方式/
                label_val[16].append('权威型教养方式')
            if label == 'label43':
                label_val[16].append('专制型教养方式')
            if label == 'label44':
                label_val[16].append('溺爱型教养方式')
            if label == 'label45':
                label_val[16].append('忽视型教养方式')

            if label == 'label46':  # 家庭气氛
                label_val[17].append('平静型家庭氛围')
            if label == 'label47':
                label_val[17].append('和谐型家庭氛围')
            if label == 'label48':
                label_val[17].append('冲突型家庭氛围')
            if label == 'label49':
                label_val[17].append('离散型家庭氛围')

            if label == 'label53':  # 成员文化程度 如果标了认为是文化程度低，不标的为空
                label_val[18].append('成员文化程度低')

            if label == 'label50':  # 成员健康状况
                label_val[19].append('父母生理疾病')
            if label == 'label51':
                label_val[19].append('父母心理疾病')

            if label == 'label54':  # 家庭经济状况
                label_val[20].append('家庭经济低收入')
            if label == 'label55':
                label_val[20].append('家庭经济高收入')

            if label == 'label52':  # 成员不良行为 如果标了认为是存在不良行为，不标的默认为空
                label_val[21].append('家庭成员存在不良行为')

            if label == 'label56':  # /教师领导方式
                label_val[22].append('教师领导方式权威型')
            if label == 'label57':
                label_val[22].append('教师领导方式民主型')
            if label == 'label58':
                label_val[22].append('教师领导方式放任型')

            if label == 'label59':  # 同伴接纳
                label_val[23].append('同伴接纳被欢迎')
            if label == 'label60':
                label_val[23].append('同伴接纳一般')
            if label == 'label61':
                label_val[23].append('同伴接纳矛盾')
            if label == 'label62':
                label_val[23].append('同伴接纳被忽视')
            if label == 'label63':
                label_val[23].append('同伴接纳被拒绝')

            if label == 'label66':  # 根本原因
                label_val[26].append('情绪容易冲动')
            if label == 'label67':
                label_val[26].append('认知方式过激')
            if label == 'label68':
                label_val[26].append('病理性不自控')
            if label == 'label69':
                label_val[26].append('安全感需求不满足')
            if label == 'label70':
                label_val[26].append('友情需求不满足')
            if label == 'label71':
                label_val[26].append('亲情需求不满足')
            if label == 'label72':
                label_val[26].append('爱情需求不满足')
            if label == 'label73':
                label_val[26].append('对他人尊重的不足')
            if label == 'label74':
                label_val[26].append('被他人尊重的需求不满足')
            if label == 'label75':
                label_val[26].append('被他人关注和重视的需求不满足')
            if label == 'label76':
                label_val[26].append('自信心不足')
            if label == 'label77':
                label_val[26].append('成就感需求不满足')
            if label == 'label78':
                label_val[26].append('认知需求不平衡')
            if label == 'label79':
                label_val[26].append('认知理解偏差')
            if label == 'label80':
                label_val[26].append('无知模仿')
            if label == 'label81':
                label_val[26].append('缺失教育引导')

            # 育人对策
            if label == 'label82':  # 根本原因
                label_val[27].append('说服教育法')
            if label == 'label83':
                label_val[27].append('情感陶冶法')
            if label == 'label84':
                label_val[27].append('榜样示范法')
            if label == 'label85':
                label_val[27].append('自我教育法')
            if label == 'label86':
                label_val[27].append('实践锻炼法')
            if label == 'label87':
                label_val[27].append('品德评价法')
            if label == 'label88':
                label_val[27].append('赏识教育法')
            if label == 'label89':
                label_val[27].append('学业帮扶法')
            if label == 'label90':
                label_val[27].append('家庭教育法')
            if label == 'label91':
                label_val[27].append('与社区合作法')
            if label == 'label92':
                label_val[27].append('志愿活动法')
            if label == 'label93':
                label_val[27].append('在家学习法')
            if label == 'label94':
                label_val[27].append('与家长沟通法')
            if label == 'label95':
                label_val[27].append('参与决策法')

    # print(label_val)

    return label_val


def save_excel(titles, labels, ID):
    '''
    将数据保存在表格中
    :param titles:
    :param labels:
    :return:
    '''

    # workbook = xlsxwriter.Workbook('part2_labels.xlsx')
    workbook = xlsxwriter.Workbook('part3_labels.xlsx')
    # workbook = xlsxwriter.Workbook('part1_labels.xlsx')
    worksheet = workbook.add_worksheet()  # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误

    row = 1
    col = 0
    startId = 174
    for k in range(len(titles)):
        # # 第二部分递增索引
        # worksheet.write(row, col, startId + row)
        # # 第一部分用原来的索引
        worksheet.write(row, col, ID[k])
        worksheet.write(row, col + 1, titles[k])
        row += 1

    row = 1
    col = 0
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            text = ''
            for tt in labels[i][j]:
                text += tt
                text += ','
            # print(text)
            if j <= 14:
                worksheet.write(row, col + 2 + j, text)
            if j > 14 and j <= 16:
                worksheet.write(row, col + 2 + 7 + j, text)
            if j > 16 and j <= 21:
                worksheet.write(row, col + 2 + 7 + 1 + j, text)
            if j > 21 and j <= 22:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + j, text)
            if j > 22 and j <= 23:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + 1 + j, text)
            if j > 23 and j <= 24:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + 1 + 1 + j, text)
            if j > 24 and j <= 27:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + 1 + 1 + j, text)
        row += 1
    workbook.close()


if __name__ == "__main__":
    titles, labels, ID = read_ann()
    # id 从175开始，然后是title，然后是label
    print(titles)
    print(labels)
    save_excel(titles, labels, ID)
    print('保存完毕 Finished！')
