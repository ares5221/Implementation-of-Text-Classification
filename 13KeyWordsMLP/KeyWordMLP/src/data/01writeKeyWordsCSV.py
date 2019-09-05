#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv

'''将数据整理为csv格式'''
# 1 all data
a1 = '身体攻击行为,打,伤人,打架,斗殴,暴力,好斗,踢打,伤害,踹,蹬,挠,拿东西怼,拿东西砸,掐,扎,推,骚扰,欺负,攻击,霸道,朝别人吐口水,朝同学泼开水,用粉笔头丢同学,破坏同学东西,撞同学,向同学扔东西,扇耳光,划伤,身体冲撞,殴打,打群架,推搡同学'
a2 = '言语攻击行为,嘲笑,骂人,言语威胁,吓唬,吵架,言语冲撞,顶撞,吼同学,言语恐吓,言语挑衅,揭短,言语凌辱,顶嘴,责骂,取笑,说脏话,抬杠,与父母大吵大闹,辱骂,起外号,嘲讽'
a3 = '关系攻击行为,诽谤,挑衅老师,敌视,漠视,回避,敌意,与老师对着干,摆脸色,故意不搭理,不听话,诋毁,对着干,说别人坏话,捉弄同学,说坏话,污蔑,讽刺,诋毁,挑拔,离间,孤立同学'
a4 = '不听讲,看课外书,摆弄东西,学习障碍,捣乱打闹,教室乱跑,扔纸团,扰乱考场纪律,不遵守学校规章制度,无视校规校纪,撒谎,说谎偷窃,抢劫' \
     '沉默寡言,不爱说话,胆小怕事,自卑,胆战心惊,矛盾,情绪低落,抑郁,悲观,易怒,偏激,焦虑,情绪不稳定,理解能力差,学习能力差,' \
     '缺乏兴趣,态度不端正,放纵,任性,多动,注意力,沉迷游戏,早恋,自伤自残,失恋'


l1 = '学习能力问题,理解能力差,学习能力差,学习障碍,记忆力差,记不住东西,忘性大,记得慢,阅读障碍,听写障碍,智力低下,上课听不懂,补课效果不好,无法理解,不能理解,学习有障碍,理解有难度,很难理解,总是理解不了,总是学不会,理解能力差,智商不高,听不懂课,理解不了上课内容,看不懂题目'
l2 = '学习方法问题,学习很努力但是成绩不提高,不读英语,不会整理,学习没有计划,学习方法不当,严重偏科,抓不住重点,学习习惯差,边学习边玩,钻牛角尖,没有学习计划,不会规划学习时间,学习效率低,不会管理学习时间,没有掌握学习方法,不开窍,方法不当,没有掌握学习技巧,死记硬背'
l3 = '学习态度问题,不想上学,想退学,不想上课,不爱学习,厌学,没有兴趣,无兴趣,缺乏兴趣,态度不端正,不完成作业,不按时交作业,不交作业,不做课后作业,拖拉作业,潦草,乱写乱画,不认真,上课不带课本,抄袭作业,不背书,抄作业,忘带作业,忘交作业,上课不回答问题,对成绩无所谓,对学习无所谓,放任自流,不想学习,讨厌学习,没有学习兴趣,不写作业,作业马虎,随意完成作业,上课似听非听,作业爱做不做,上课一个字也听不进去,交白卷,不参加考试'
l4 = '注意力问题,注意力不集中,不集中,注意力保持时间短,分心,犯粗心错误,马虎,多动,出神,写错别字,注意力,无法专注做事,走神,发呆,开小差,注意力分散,注意力不持久'
l5 = '打,伤人,打架,骂人,言语威胁,吓唬,吵架,诽谤,挑衅老师,敌视,漠视上课小动作,捣乱打闹,教室乱跑,' \
     '扰乱课堂秩序,扰乱考场纪律,不遵守学校规章制度,无视校规校纪,撒谎,说谎偷窃,抢劫' \
     '沉默寡言,不爱说话,胆小怕事,自卑,胆战心惊,矛盾,情绪低落,抑郁,悲观,易怒,偏激,焦虑,情绪不稳定,理解能力差,学习能力差,' \
     '缺乏兴趣,态度不端正,放纵,任性,多动,注意力,沉迷游戏,早恋,自伤自残,失恋'

# 2 构建攻击行为关键词的标注文件
body_attack = a1.split(',')
verbal_attack = a2.split(',')
relationship_attack = a3.split(',')
no_attack = a4.split(',')
all_attack_words = body_attack + verbal_attack + relationship_attack + no_attack
print(len(body_attack), len(verbal_attack), len(relationship_attack), len(no_attack), len(all_attack_words)/4)
# # 2.1将攻击行为相关词保存为csv文件
with open("all_attack_words.csv", "a", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    count_attack = 0
    for first in all_attack_words:
        for second in all_attack_words:
            if first == second:
                continue
            else:
                count_attack += 1
                if first in body_attack and second in body_attack:
                    label = 1
                elif first in verbal_attack and second in verbal_attack:
                    label = 1
                elif first in relationship_attack and second in relationship_attack:
                    label = 1
                elif first in no_attack and second in no_attack:
                    label = 1
                else:
                    label = 0
            # print(first, second, label)
            writer.writerow([first, second, label])
    print('---将攻击行为关键词写入csv文件完成 共得到%d 条标记数据---' % count_attack)
#
# # 3 构建学习问题关键词的标注文件
learning_ability = l1.split(',')
learning_method = l2.split(',')
learning_attitude = l3.split(',')
learning_attention = l4.split(',')
learning_no = l5.split(',')
all_learn_words = learning_ability + learning_method + learning_attitude + learning_attention + learning_no
print(len(learning_ability), len(learning_method), len(learning_attitude), len(learning_attention), len(learning_no), len(all_learn_words)/5)
# # 3.1将学习问题相关词保存为csv文件
with open("all_learn_words.csv", "a", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    count_learn = 0
    for first in all_learn_words:
        for second in all_learn_words:
            if first == second:
                continue
            else:
                count_learn += 1
                if first in learning_ability and second in learning_ability:
                    label = 1
                elif first in learning_method and second in learning_method:
                    label = 1
                elif first in learning_attitude and second in learning_attitude:
                    label = 1
                elif first in learning_attention and second in learning_attention:
                    label = 1
                elif first in learning_no and second in learning_no:
                    label = 1
                else:
                    label = 0
            # print(first, second, label)
            writer.writerow([first, second, label])
    print('---将学习问题关键词写入csv文件完成 共得到%d 条标记数据---' % count_learn)
