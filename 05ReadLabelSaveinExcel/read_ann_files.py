#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import re

'''读取标注文件.ann,将其中label信息保存在excel中'''

def read_ann():
    '''
    读取文件夹下ann文本的内容，读取其中label
    :return: texts
    '''
    annFilePath = os.path.abspath('../dataset/AnnFileForLabel/cases')
    print('待比较文档路径：', annFilePath)
    titles = []
    doc_num = 0
    for fname in os.listdir(annFilePath):
        if fname[-4:] == '.txt':
            f = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            tmp = f.readline() # 读取第一行为title
            print('id--->', doc_num, ' title--->', tmp)
            titles.append(tmp)
            doc_num +=1
            f.close()
        if fname[-4:] == '.ann':
            ff = open(os.path.join(annFilePath, fname), 'r', encoding='UTF-8')
            tmp = ff.readlines()
            print('@@@@@@@@@@@@@@', tmp)
            convert_label2value(tmp)
            ff.close()
    print('读取到', doc_num, '篇文档')


def convert_label2value(label_list):
    len_data = len(label_list)
    label_val = [[] for i in range(42)]
    label_val[14] = '一般'

    for i in range(len_data):
        label_value = label_list[i]
        if i <13:
            print(i, label_value)
            p = re.compile(r'label\d{1,2}')
            q = re.compile(r'.+\w')
            print('ss',p.findall(label_value))
            # print('ssss',q.findall(label_value.split('\t')[2]))
            label = p.findall(label_value)[0]
            val = q.findall(label_value.split('\t')[2])[0]


            print(label, type(label), val, type(val))
            #前面处理基础信息，需要正则表达式规格化数据
            if label == 'label31':  # 年级
                print(val)
                label_val[11] = val
            if label == 'label32':  # 性别
                print(val)
                label_val[10] = val
            if label == 'label33':  # 年龄
                label_val[12] = val

            if label == 'label64':  # 大众传媒影响
                label_val[24] = val
            if label == 'label65':  # 社会文化社会风气
                label_val[25] = val

            if label == 'label1':  # 攻击行为/
                label_val[0].append('身体攻击行为')
            if label == 'label2':
                label_val[0].append( '言语攻击')
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

            if label == 'label26':   # 自我中心/
                label_val[7].append('自我吹嘘型问题')
            if label == 'label27':
                label_val[7].append('执拗型问题')
            if label == 'label28':
                label_val[7].append('自私型问题')

            if label == 'label22':    # 学习问题/
                label_val[8].append('学习能力问题')
            if label == 'label23':
                label_val[8].append('学习方法问题')
            if label == 'label24':
                label_val[8].append('学习态度问题')
            if label == 'label25':
                label_val[8].append('注意力问题')

            if label == 'label29': # 极端事件/
                label_val[9].append('青春期问题')
            if label == 'label30':
                label_val[9].append('极端问题')

            if label == 'label34':  # 健康状况/
                label_val[13].append('生理疾病')
            if label == 'label35':
                label_val[13].append('心理疾病')

            if label == 'label36':    # 所属群体/默认 一般
                label_val[14].append('留守儿童')
            if label == 'label37':
                label_val[14].append('流动儿童')
            if label == 'label38':
                label_val[14].append('孤困儿童')

            if label == 'label39':    # 家庭结构/
                label_val[15].append('寄养家庭')
            if label == 'label40':
                label_val[15].append('重组家庭')
            if label == 'label41':
                label_val[15].append('单亲家庭')

            if label == 'label42':    # 教养方式/
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

            if label == 'label56':    # /教师领导方式
                label_val[22].append('教师领导方式权威型')
            if label == 'label57':
                label_val[22].append('教师领导方式民主型')
            if label == 'label58':
                label_val[22].append('教师领导方式放任型')

            if label == 'label59':    # 同伴接纳
                label_val[23].append('同伴接纳被欢迎')
            if label == 'label60':
                label_val[23].append('同伴接纳一般')
            if label == 'label61':
                label_val[23].append('同伴接纳矛盾')
            if label == 'label62':
                label_val[23].append('同伴接纳被忽视')
            if label == 'label63':
                label_val[23].append('同伴接纳被拒绝')

            if label == 'label66':    # 根本原因
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

            print(label_val)


if __name__ == "__main__":
    texts = read_ann()
    #id 从175开始，然后是title，然后是label