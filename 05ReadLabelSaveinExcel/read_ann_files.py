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
    #攻击行为/违纪行为/品行问题/不良嗜好/退缩/抑郁问题/焦虑问题/自我中心/学习问题/极端事件/
    # 性别/年级/健康状况/所属群体/家庭结构/教养方式/家庭气氛/成员文化程度/成员健康状况
    # /家庭经济状况/成员不良行为/教师领导方式/教师教学风格/同伴接纳/大众传媒/社会文化社会风气/原因/对策

    label_val = ['' for i in range(42)]
    value_dic = {'label1': '身体攻击行为', 'label2': '言语攻击', 'label3': '间接攻击',
                 'label4': '隐蔽性违纪行为', 'label5': '违反课堂制度行为', 'label6': '扰乱课堂秩序行为', 'label7': '违反校规校级行为',
                 'label8': '欺骗行为', 'label9': '偷盗行为', 'label10': '背德行为',
                 'label11': '强迫行为', 'label12': '沉迷行为',
                 'label13': '言语性退缩', 'label14': '行为性退缩', 'label15': '心理性退缩',
                 'label16': '抑郁情绪问题', 'label17': '抑郁行为问题', 'label18': '抑郁思维问题',
                 'label19': '焦虑情绪问题', 'label20': '焦虑行为问题', 'label21': '焦虑思维问题',
                 'label22': '学习能力问题', 'label23': '学习方法问题', 'label24': '学习态度问题', 'label25': '注意力问题',
                 'label26': '自我吹嘘型问题', 'label27': '执拗型问题', 'label28': '自私型问题',
                 'label29': '青春期问题', 'label30': '极端问题',
                 'label31': '年级',
                 'label32': '性别',
                 'label33': '年龄',
                 'label34': '生理疾病', 'label35': '心理疾病',
                 'label36': '留守儿童', 'label37': '流动儿童', 'label38': '孤困儿童',  # 默认一般
                 'label39': '寄养家庭', 'label40': '重组家庭', 'label41': '单亲家庭',
                 'label42': '权威型教养方式', 'label43': '专制型教养方式', 'label44': '溺爱型教养方式', 'label45': '忽视型教养方式',
                 'label46': '平静型家庭氛围', 'label47': '和谐型家庭氛围', 'label48': '冲突型家庭氛围', 'label49': '离散型家庭氛围',
                 'label50': '父母生理疾病', 'label51': '父母心理疾病',
                 'label52': '成员不良行为',  # 有或者无
                 'label53': '成员文化程度',
                 'label54': '家庭经济低收入', 'label55': '家庭经济高收入',
                 'label56': '教师领导方式权威型', 'label57': '教师领导方式民主型', 'label58': '教师领导方式放任型',
                 'label59': '同伴接纳被欢迎', 'label60': '同伴接纳一般', 'label61': '同伴接纳矛盾', 'label62': '同伴接纳被忽视', 'label63': '同伴接纳被拒绝',
                 'label64': '大众传媒影响',
                 'label65': '社会风气影响',
                    #原因
                 'label66': '情绪容易冲动',
                 'label67': '认知方式过激',
                 'label68': '病理性不自控',
                 'label69': '安全感需求不满足',
                 'label70': '友情需求不满足',
                 'label71': '亲情需求不满足',
                 'label72': '爱情需求不满足',
                 'label73': '对他人尊重的不足',
                 'label74': '被他人尊重的需求不满足',
                 'label75': '被他人关注和重视的需求不满足',
                 'label76': '自信心不足',
                 'label77': '成就感需求不满足',
                 'label78': '认知需求不平衡',
                 'label79': '认知理解偏差',
                 'label80': '无知模仿',
                 'label81': '缺失教育引导',
                #解决办法
                 'label82': '说服教育法',
                 'label83': '情感陶冶法',
                 'label84': '榜样示范法',
                 'label85': '自我教育法',
                 'label86': '实践锻炼法',
                 'label87': '品德评价法',
                 'label88': '赏识教育法',
                 'label8': '学业帮扶法',
                 'label90': '家庭教育法',
                 'label91': '与社区合作法',
                 'label92': '志愿活动法',
                 'label93': '在家学习法',
                 'label94': '与家长沟通法',
                 'label95': '参与决策法'

                 }
    for i in range(len_data):
        label_value = label_list[i]
        if i <1:
            print(i, label_value)
            p = re.compile(r'label\d{1,2}')
            q = re.compile(r'.+\w')
            # print('sss',p.findall(label_value))
            # print('sss',q.findall(label_value.split('\t')[2]))
            label = p.findall(label_value)
            val = q.findall(label_value.split('\t')[2])
            #前面处理年龄性别年级信息，需要正则表达式规格化数据
            # if label == 'label1':
            #     label_value[0] = val
            # elif label == 'label2':
            #     label_value[0] = val
            # elif label == 'label3':
            #     label_value[0] = val
            # elif label == 'label4':
            #     label_value[1] = val

            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label2')
            label_value[0] = value_dic.get(label, 'label3')
            label_value[1] = value_dic.get(label, 'label4')
            label_value[1] = value_dic.get(label, 'label5')
            label_value[1] = value_dic.get(label, 'label6')
            label_value[1] = value_dic.get(label, 'label7')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')
            label_value[0] = value_dic.get(label, 'label1')





if __name__ == "__main__":
    texts = read_ann()
    #id 从175开始，然后是title，然后是label