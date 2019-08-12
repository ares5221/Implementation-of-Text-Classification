#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re
import xlsxwriter  # write xlsx   # xlwt 只能写xls

'''
读取标注文件.ann,将其中title，ID,label信息保存在excel中
采用了新版的label
'''


def read_ann():
    '''
    :return: titles, labels
    '''
    # 新的第三部分标注数据385
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
            titles.append(title)
            doc_num += 1
            text_index.append(fname[:-4])
            f.close()
        if fname[-4:] == '.ann':
            ff = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            label_list = ff.readlines()
            label = convert_label2value(label_list)
            print('name--->', fname, 'label--->', label)
            ss = []  # 去除重复的标签
            for s_label in label:
                new_label_val = list(set(s_label))
                ss.append(new_label_val)
            labels.append(ss)
            ff.close()
    print('读取到', doc_num, '篇文档', len(titles), len(labels))
    return titles, labels, text_index


def formatting_grade_info(grade_info):
    grade_formatting = grade_info
    primary_grade = ['一', '二', '三', '四', '五', '六']
    if grade_info in primary_grade:
        grade_formatting = grade_info + '年级'
    if grade_info == '七' or grade_info == '七年级' or grade_info == '7':
        grade_formatting = '初一'
    if grade_info == '八' or grade_info == '八年级' or grade_info == '8':
        grade_formatting = '初二'
    if grade_info == '九' or grade_info == '九年级' or grade_info == '9':
        grade_formatting = '初三'
    return grade_formatting


def convert_label2value(label_list):
    label_val = [[] for i in range(27)]
    for i in range(len(label_list)):
        label_str = label_list[i]
        if True:
            p = re.compile(r'label[A-Za-z0-9]+')  # 正则表达式提取label[A-Za-z0-9]+
            label_str_list = label_str.split('\t')
            label = p.findall(label_str_list[1])[0]
            val = label_str_list[2].replace('\n', '')
            # labela | 基本信息
            if label == 'label2':  # 性别
                if '她' in val or '女' in val:
                    val = '女'
                else:
                    val = '男'
                label_val[8] = [val]
            if label == 'label1':  # 年级
                if '学生' in val:
                    val = val.replace('学生', '')
                val = formatting_grade_info(val)
                label_val[9] = [val]

            if label == 'label3':  # 年龄
                if '年龄' in val:
                    val = val.replace('年龄', '')
                    if '：' in val:
                        val = val.replace('：', '')
                        if '，' in val:
                            val = val.replace('，', '')
                label_val[10] = [val]

            if label == 'label4':  # 健康状况
                label_val[11].append('健康')
            if label == 'label5':
                label_val[11].append('生理疾病')
            if label == 'label6':
                label_val[11].append('心理疾病')

            if label == 'label7':  # 所属群体/默认 一般
                label_val[12] = ['一般儿童']
            if label == 'label8':
                label_val[12] = ['留守儿童']
            if label == 'label9':
                label_val[12] = ['流动儿童']
            if label == 'label10':
                label_val[12] = ['孤困儿童']

            # labelb | 问题行为
            if label == 'label11':  # 攻击行为/
                label_val[0].append('身体攻击行为')
            if label == 'label12':
                label_val[0].append('言语攻击行为')
            if label == 'label13':
                label_val[0].append('关系攻击行为')

            if label == 'label14':  # 违纪行为/
                label_val[1].append('隐蔽性违反课堂纪律行为')
            if label == 'label15':
                label_val[1].append('扰乱课堂秩序行为')
            if label == 'label16':
                label_val[1].append('违反课外纪律行为')

            if label == 'label17':  # 不良行为/
                label_val[2].append('欺骗行为')
            if label == 'label18':
                label_val[2].append('偷盗行为')
            if label == 'label19':
                label_val[2].append('背德行为')

            if label == 'label20':  # 社会退缩/
                label_val[3].append('言语型退缩')
            if label == 'label21':
                label_val[3].append('行为型退缩')
            if label == 'label22':
                label_val[3].append('心理型退缩')

            if label == 'label23':  # 情绪问题/
                label_val[4].append('抑郁问题')
            if label == 'label24':
                label_val[4].append('焦虑问题')

            if label == 'label29':  # 自我中心/
                label_val[5].append('自我吹嘘型问题')
            if label == 'label30':
                label_val[5].append('执拗型问题')
            if label == 'label31':
                label_val[5].append('自私型问题')

            if label == 'label25':  # 学习问题/
                label_val[6].append('学习能力问题')
            if label == 'label26':
                label_val[6].append('学习方法问题')
            if label == 'label27':
                label_val[6].append('学习态度问题')
            if label == 'label28':
                label_val[6].append('注意力问题')

            if label == 'label32':
                label_val[7].append('沉迷行为')
            if label == 'label33':
                label_val[7].append('早恋行为')
            if label == 'label34':
                label_val[7].append('极端行为')

            # labelc | 家庭因素
            if label == 'label35':  # 家庭结构/
                label_val[13].append('寄养家庭')
            if label == 'label36':
                label_val[13].append('重组家庭')
            if label == 'label37':
                label_val[13].append('单亲家庭')
            if label == 'label38':
                label_val[13].append('完整家庭')

            if label == 'label39':  # 教养方式/
                label_val[14].append('权威型教养方式')
            if label == 'label40':
                label_val[14].append('专制型教养方式')
            if label == 'label41':
                label_val[14].append('溺爱型教养方式')
            if label == 'label42':
                label_val[14].append('忽视型教养方式')

            if label == 'label43':  # 家庭气氛
                label_val[15].append('平静型家庭氛围')
            if label == 'label44':
                label_val[15].append('和谐型家庭氛围')
            if label == 'label45':
                label_val[15].append('冲突型家庭氛围')
            if label == 'label46':
                label_val[15].append('离散型家庭氛围')

            if label == 'label50':
                label_val[16].append('成员文化程度高')
            if label == 'label51':
                label_val[16].append('成员文化程度低')

            if label == 'label47':  # 成员健康状况
                label_val[17].append('成员生理疾病')
            if label == 'label48':
                label_val[17].append('成员心理疾病')
            if label == 'label49':
                label_val[17].append('成员健康')

            if label == 'label52':  # 家庭经济状况
                label_val[18].append('家庭经济低收入')
            if label == 'label53':
                label_val[18].append('家庭经济高收入')

            if label == 'label54':  # 成员不良行为 如果标了认为是存在不良行为，不标的默认为空
                label_val[19].append('成员不良行为')

            # labeld | 学校因素
            if label == 'label55':  # /教师领导方式
                label_val[20].append('教师领导方式权威型')
            if label == 'label56':
                label_val[20].append('教师领导方式民主型')
            if label == 'label57':
                label_val[20].append('教师领导方式放任型')

            if label == 'label58':  # 同伴接纳
                label_val[21].append('同伴接纳受欢迎')
            if label == 'label59':
                label_val[21].append('同伴接纳一般型')
            if label == 'label60':
                label_val[21].append('同伴接纳矛盾型')
            if label == 'label61':
                label_val[21].append('同伴接纳被忽视')
            if label == 'label62':
                label_val[21].append('同伴接纳被拒绝')

            if label == 'label63':  # 大众传媒影响
                # label_val[22].append('大众传媒的影响')
                label_val[22] = [val]

            if label == 'label64':  # 社会文化社会风气
                label_val[23].append('社会风气读书无用')
            if label == 'label65':  # 社会文化社会风气
                label_val[23].append('社会风气重男轻女')

            if label == 'label66':  # 根本原因
                label_val[24].append('情绪控制能力差')
            if label == 'label67':
                label_val[24].append('病理性缺陷')
            if label == 'label68':
                label_val[24].append('缺乏安全感')
            if label == 'label69':
                label_val[24].append('缺乏友情支持')
            if label == 'label70':
                label_val[24].append('缺乏亲情关爱')
            if label == 'label71':
                label_val[24].append('恋爱关系受挫')
            if label == 'label72':
                label_val[24].append('被他人关注需求不满足')
            if label == 'label73':
                label_val[24].append('自尊心受挫')
            if label == 'label74':
                label_val[24].append('缺乏自信')
            if label == 'label75':
                label_val[24].append('认知需求不平衡')
            if label == 'label76':
                label_val[24].append('错误的认知理解')
            if label == 'label77':
                label_val[24].append('缺少正确教育引导')

            # labelf | 教育学生的具体方法
            if label == 'label78':
                label_val[25].append('与学生交流谈心')
            if label == 'label79':
                label_val[25].append('把握时机引导学生')
            if label == 'label80':
                label_val[25].append('多途径与学生沟通')
            if label == 'label81':
                label_val[25].append('树立教师榜样力量')
            if label == 'label82':
                label_val[25].append('发挥同伴榜样力量')
            if label == 'label83':
                label_val[25].append('发挥故事人物榜样力量')
            if label == 'label84':
                label_val[25].append('用自身人格感化学生')
            if label == 'label85':
                label_val[25].append('凝聚集体力量共同帮扶')
            if label == 'label86':
                label_val[25].append('给予学生关怀、爱护和尊重')
            if label == 'label87':
                label_val[25].append('创设情境熏陶教学')
            if label == 'label88':
                label_val[25].append('通过多种教学活动引导学生')
            if label == 'label89':
                label_val[25].append('通过各方面的道德要求激发学生反省')
            if label == 'label90':
                label_val[25].append('引导学生监控评价自身道德表现')
            if label == 'label91':
                label_val[25].append('提醒学生遵守规章制度')
            if label == 'label92':
                label_val[25].append('委托班级任务')
            if label == 'label93':
                label_val[25].append('鼓励参加实践活动')
            if label == 'label94':
                label_val[25].append('给予学生展现的机会')
            if label == 'label95':
                label_val[25].append('及时鼓励进步')
            if label == 'label96':
                label_val[25].append('及时批评错误')
            if label == 'label97':
                label_val[25].append('进行阶段性全面评价')
            if label == 'label98':
                label_val[25].append('积极发现学生闪光点')
            if label == 'label99':
                label_val[25].append('鼓励同学发现闪光点')
            if label == 'label100':
                label_val[25].append('帮助学生补课')
            if label == 'label101':
                label_val[25].append('成立学习帮扶小组')
            if label == 'label102':
                label_val[25].append('为学生制定学习计划和方案')
            if label == 'label103':
                label_val[25].append('为学生创造有利的学习环境')
            # labelg | 家校合作的具体方法
            if label == 'label104':
                label_val[25].append('帮助家长创造家庭教育环境')
            if label == 'label105':
                label_val[25].append('传授家庭教育知识')
            if label == 'label106':
                label_val[25].append('鼓励家长以身作则')
            if label == 'label107':
                label_val[25].append('协调社区力量为学生提供服务')
            if label == 'label108':
                label_val[25].append('招收家长作为志愿者参与学校活动')
            if label == 'label109':
                label_val[25].append('引导家长和学生共同学习')
            if label == 'label110':
                label_val[25].append('鼓励共同制定作息计划')
            if label == 'label111':
                label_val[25].append('家校互通形成合力')
            if label == 'label112':
                label_val[25].append('告知家长学生的在校情况')
            if label == 'label113':
                label_val[25].append('沟通了解学生的在家情况')
            if label == 'label114':
                label_val[25].append('让家长参与学校的决策管理')

            if label == 'labelh':
                label_val[26].append('已经删除')

    return label_val


def save_excel(titles, labels, ID):
    '''
    将数据保存在表格中
    :param titles:
    :param labels:
    :return:
    '''
    # workbook = xlsxwriter.Workbook('part1_labels.xlsx')
    # workbook = xlsxwriter.Workbook('part2_labels.xlsx')
    workbook = xlsxwriter.Workbook('part3_labels.xlsx')

    worksheet = workbook.add_worksheet()

    row = 1
    col = 0
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
            for k in range(len(labels[i][j])):
                if k < len(labels[i][j]) -1:
                    text += labels[i][j][k]
                    text += '，'
                else:
                    text += labels[i][j][k]
            print(text)
            if j <= 12:
                worksheet.write(row, col + 2 + j, text)
            if j > 12 and j <= 14:
                worksheet.write(row, col + 2 + 7 + j, text)
            if j > 14 and j <= 19:
                worksheet.write(row, col + 2 + 7 + 1 + j, text)
            if j > 19 and j <= 20:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + j, text)
            if j > 20 and j <= 21:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + 1 + j, text)
            if j > 21 and j <= 26:
                worksheet.write(row, col + 2 + 7 + 1 + 3 + 1 + 1 + j, text)
        row += 1
    workbook.close()


if __name__ == "__main__":
    titles, labels, ID = read_ann()
    save_excel(titles, labels, ID)
    print('保存完毕 Finished！')
