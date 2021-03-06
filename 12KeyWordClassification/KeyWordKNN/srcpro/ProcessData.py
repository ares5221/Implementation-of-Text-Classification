#!/usr/bin/env python
# _*_ coding:utf-8 _*_

def process_data():
    '''
    :return: texts,labels
    '''
    data = [[], [], [], []]
    data[0] = [[], [], []]  # 身体攻击行为 言语攻击行为 关系攻击行为
    data[1] = [[], [], []]  # 隐蔽性违反课堂纪律行为 扰乱课堂秩序行为 违反课外纪律行为
    data[2] = [[], [], []]  # 言语型退缩 行为型退缩 心理型退缩
    data[3] = [[], [], [], []]  # 学习能力问题 学习方法问题 学习态度问题 注意力问题

    data[0][0] = ['打人', '伤人', '打架', '斗殴', '暴力倾向', '好斗', '踢打别人', '伤害同学', '骚扰同学', '欺负同学', '攻击同学', '横行霸道', '朝别人吐口水',
                  '朝同学泼开水', '用粉笔头丢同学', '破坏同学东西']
    data[0][1] = ['骂人', '威胁同学', '吓唬同学', '吵架', '与家长顶嘴', '冲撞老师', '顶撞老师', '辱骂老师']
    data[0][2] = ['诽谤', '恶作剧', '捉弄别人', '嘲笑同学', '挑衅老师', '敌视老师', '漠视老师', '回避老师', '对老师抱有敌意', '与老师发生强烈冲突', '不听劝导', '叛逆',
                  '与老师对着干']

    data[1][0] = ['上课睡觉', '上课盘腿坐', '上课发呆走神', '不想进教室', '上课东张西望', '上课不听讲', '上课看课外书', '上课吃东西']
    data[1][1] = ['上课小动作', '擅自下位', '上课捣乱打闹', '教室乱跑', '上课突然往外跑', '扰乱课堂秩序', '违反自习纪律', '上课自言自语',
                  '上课随意讲话', '上课接话茬', '发出怪声', '上课嬉笑怪叫', '扰乱别人学习', '上课玩飞机', '上课扔纸团', '上课做鬼脸', '上课乱跑乱叫', '地上打滚']
    data[1][2] = ['破坏公物', '打碎班级门玻璃', '作弊', '带危险品去学校', '不穿校服', '扰乱考场纪律', '不遵守学校规章制度', '无视校规校纪', '吸烟', '喝酒', '逃学', '旷课',
                  '迟到', '奇装异服', '过分打扮']

    data[2][0] = ['沉默寡言', '少言寡语', '说话声音小', '不爱说话', '在人前说话紧张', '很少与同伴交流', '不和别人交流']
    data[2][1] = ['不想有同桌', '回避别人的眼神', '独自玩耍', '不理人', '难交朋友', '回避他人', '从不主动举手', '不与同伴交往', '拒绝集体活动', '不合群', '落寞寡群',
                  '办事退缩', '孤独']
    data[2][2] = ['胆小怕事', '自卑', '胆战心惊', '矛盾', '胆小', '拘谨', '害羞', '冷漠']

    data[3][0] = ['理解能力差', '学习能力差', '学习障碍']
    data[3][1] = ['学习基础差', '学习很努力但是成绩不提高', '不读英语', '不会整理', '学习没有计划', '学习方法不当', '严重偏科', '学习成绩下降', '学习成绩差']
    data[3][2] = ['找各种理由不想上学', '不想上学', '想退学', '不想上课', '不爱学习', '厌学', '对学习没有兴趣', '学习无兴趣', '学习缺乏兴趣',
                  '学习态度不端正', '不完成作业', '不按时交作业', '不交作业', '不做课后作业', '拖拉作业', '作业潦草', '乱写乱画', '作业不认真', '书写潦草', '字迹潦草',
                  '上课不带课本', '抄袭作业', '学习不专心']
    data[3][3] = ['注意力不集中', '精力不集中', '注意力保持时间短', '容易分心', '犯粗心错误', '马虎', '多动', '上课开小差']

    # print(data[0])
    # print(data[1])
    # print(data[2])
    # print(data[3])
    content, tag = [], []
    count = 0
    for k in range(len(data)):
        for i in range(len(data[k])):
            for j in range(len(data[k][i])):
                content.append(data[k][i][j])
                tag.append(count)
            count +=1

    classes_num = count
    return content, tag, classes_num



if __name__ == '__main__':
    content, tag, classes_num = process_data()
    print(len(content), content)
    print(len(tag), tag)
    print('获取数据完成', classes_num)
