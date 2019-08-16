#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv, os
from bert_serving.client import BertClient

'''
将攻击行为及学习问题的关键词通过bert生成每个词的embedding向量
便于在调用的时候可以直接导入
生成的数据保存在test_attack_words.npy  test_learn_words.npy
'''

# 1 all data
a1 = '身体攻击行为,打,伤人,打架,斗殴,暴力,好斗,踢打,伤害,踹,蹬,挠,拿东西怼,拿东西砸,掐,扎,推,骚扰,欺负,攻击,霸道,朝别人吐口水,朝同学泼开水,用粉笔头丢同学,破坏同学东西,撞同学,向同学扔东西,扇耳光,划伤,身体冲撞,殴打,打群架,推搡同学'
a2 = '言语攻击行为,嘲笑,骂人,言语威胁,吓唬,吵架,言语冲撞,顶撞,吼同学,言语恐吓,言语挑衅,揭短,言语凌辱,顶嘴,责骂,取笑,说脏话,抬杠,与父母大吵大闹,辱骂,起外号,嘲讽'
a3 = '关系攻击行为,诽谤,挑衅老师,敌视,漠视,回避,敌意,与老师对着干,摆脸色,故意不搭理,不听话,诋毁,对着干,说别人坏话,捉弄同学,说坏话,污蔑,讽刺,诋毁,挑拔,离间,孤立同学'
a4 = '学习能力问题,理解能力差,学习能力差,学习障碍,记忆力差,记不住东西,忘性大,记得慢,阅读障碍,听写障碍,智力低下,上课听不懂,补课效果不好,无法理解,不能理解,学习有障碍,理解有难度,很难理解,总是理解不了,总是学不会,理解能力差,智商不高,听不懂课,理解不了上课内容,看不懂题目'

l1 = '学习能力问题,理解能力差,学习能力差,学习障碍,记忆力差,记不住东西,忘性大,记得慢,阅读障碍,听写障碍,智力低下,上课听不懂,补课效果不好,无法理解,不能理解,学习有障碍,理解有难度,很难理解,总是理解不了,总是学不会,理解能力差,智商不高,听不懂课,理解不了上课内容,看不懂题目'
l2 = '学习方法问题,学习很努力但是成绩不提高,不读英语,不会整理,学习没有计划,学习方法不当,严重偏科,抓不住重点,学习习惯差,边学习边玩,钻牛角尖,没有学习计划,不会规划学习时间,学习效率低,不会管理学习时间,没有掌握学习方法,不开窍,方法不当,没有掌握学习技巧,死记硬背'
l3 = '学习态度问题,不想上学,想退学,不想上课,不爱学习,厌学,没有兴趣,无兴趣,缺乏兴趣,态度不端正,不完成作业,不按时交作业,不交作业,不做课后作业,拖拉作业,潦草,乱写乱画,不认真,上课不带课本,抄袭作业,不背书,抄作业,忘带作业,忘交作业,上课不回答问题,对成绩无所谓,对学习无所谓,放任自流,不想学习,讨厌学习,没有学习兴趣,不写作业,作业马虎,随意完成作业,上课似听非听,作业爱做不做,上课一个字也听不进去,交白卷,不参加考试'
l4 = '注意力问题,注意力不集中,不集中,注意力保持时间短,分心,犯粗心错误,马虎,多动,出神,写错别字,注意力,无法专注做事,走神,发呆,开小差,注意力分散,注意力不持久'
l5 = '言语攻击行为,嘲笑,骂人,言语威胁,吓唬,吵架,言语冲撞,顶撞,吼同学,言语恐吓,言语挑衅,揭短,言语凌辱,顶嘴,责骂,取笑,说脏话,抬杠,与父母大吵大闹,辱骂,起外号,嘲讽,诽谤,挑衅老师,敌视'


def getTestNPYData(data, name):
    X = [[] for i in range(len(data))]
    bc = BertClient()
    for index in range(len(data)):
        vector = bc.encode([data[index]])
        X[index] = vector.tolist()
        if index % 100 == 0:
            print(index, 'is finish')
    train_data = np.array(X)
    Xnpyname = name + '_' + str(len(data)) + '.npy'
    np.save(Xnpyname, train_data)
    print('%s 的数据通过bert embedding 为向量npy格式' % name)


def getLabelTestNPYData(data,name):
    '''生成分类的标签信息'''
    print(len(data))
    if len(data) == 4 and name == 'label_attack_words':
        print(len(data[0]), len(data[1]), len(data[2]))
        label_attack = [0 for i in range(len(data[0]))] +[1 for i in range(len(data[1]))] +[2 for i in range(len(data[2]))]+[3 for i in range(len(data[3]))]
        np.save(name + '_' + str(102) + '.npy', label_attack)
    if len(data) == 5 and name == 'label_learn_words':
        print(len(data[0]), len(data[1]), len(data[2]), len(data[3]))
        label_learn = [0 for i in range(len(data[0]))] +[1 for i in range(len(data[1]))] +[2 for i in range(len(data[2]))] + [3 for i in range(len(data[3]))]+ [4 for i in range(len(data[4]))]
        np.save(name + '_' + str(126) + '.npy', label_learn)


# Start Position--->>>>>>>>>
if __name__ == '__main__':
    all_attack_words = a1.split(',') + a2.split(',') + a3.split(',') + a4.split(',')
    all_learn_words = l1.split(',') + l2.split(',') + l3.split(',') + l4.split(',') + l5.split(',')
    getTestNPYData(all_attack_words, 'attack_words')
    getLabelTestNPYData([a1.split(','), a2.split(','), a3.split(','), a4.split(',')], 'label_attack_words')

    getTestNPYData(all_learn_words, 'learn_words')
    getLabelTestNPYData([l1.split(','), l2.split(','), l3.split(','), l4.split(','), l5.split(',')], 'label_learn_words')