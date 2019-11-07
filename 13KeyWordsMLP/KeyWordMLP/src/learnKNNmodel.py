#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json
import os
from bert_serving.client import BertClient
import numpy as np
import tensorflow as tf

# DIR = 'semi/nlp_model/'
DIR = './data/'

def manage(sentence):
    if len(sentence) >0:
        classification_tag = KNN_learning(sentence)
    else:
        print('invalid sentence input!!')
        return
    return classification_tag


def calSimilarityByLearningModel(train_data, test_vec, topK=5):
    learning_rate = 0.001
    batch_size = 1# Network Parameters
    n_input = 1536  # Number of feature
    n_hidden_1 = 32  # 1st layer number of features
    n_classes = 2  # Number of classes to predict
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    def multilayer_perceptron(x, weights, biases):# Create model
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        model_file = tf.train.latest_checkpoint('ckpt_learning/')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        simlist, idlist = [0 for i in range(topK)], [0 for i in range(topK)]
        for i in range(len(train_data)):
            testbatch = [[] for i in range(1)]
            testbatch[0] = np.append(train_data[i], test_vec).tolist()
            res = sess.run(pred, feed_dict={x: testbatch})[0][0]
            # print('################################',res, i)
            if res > min(simlist):
                simlist[simlist.index(min(simlist))] = res
                idlist[simlist.index(min(simlist))] = i
    return idlist, simlist



def train_knnmodel(train_data, train_label, sentence):
    '''调用训练好的攻击行为 关键词的mlp模型,找到其中所属最多的label'''
    topK = 5  # K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    tf.reset_default_graph()
    idlist, simlist =calSimilarityByLearningModel(train_data, test_vec, topK)
    # 根据找到的最相似关键词的索引 获取其对应的label信息,判断其具体属于那个二级类别如身体攻击还是言语攻击
    print('通过MLP计算 输入语句( %s )与所有关键词的相似度前%d个:'%(sentence,topK), simlist)
    # print('最相似词的对应索引index:', idlist)
    get_real_word(idlist)
    cal_lab = []
    for idx in idlist:
        cal_lab.append(train_label[idx])
    sent_predict = max(cal_lab, key=cal_lab.count)
    # print(train_data[idx])
    # todo 这里显示当前找到的最相似词，
    return sent_predict


def get_real_word(most_index):
    index_2_keyword_name = './data/learn_words_index_2_keyword.json'
    ss = []
    with open(index_2_keyword_name, 'r', encoding='utf-8') as f:
        word_index_dic = json.load(f)
        for index in most_index:
            ss.append(word_index_dic[str(index)])
    print('--->最相似的词是：', ss)


def KNN_learning_pro(sentence):
    if os.path.exists(DIR + "learn_words_140.npy") and os.path.exists(DIR + "label_learn_words_145.npy"):
        train_data = np.load(DIR + "learn_words_140.npy")
        train_label = np.load(DIR + "label_learn_words_145.npy")
    else:
        pass # 这里会调用生成label_learn_words_126.npy的方法
    res = train_knnmodel(train_data, train_label, sentence)
    return res


def KNN_learning(sentence):
    if os.path.exists(DIR + "learn_words_140.npy") and os.path.exists(DIR + "label_learn_words_145.npy"):
        train_data = np.load(DIR + "learn_words_140.npy")
        train_label = np.load(DIR + "label_learn_words_145.npy")
    else:
        pass # 这里会调用生成label_learn_words_126.npy的方法
    res = train_knnmodel(train_data, train_label, sentence)
    if res == 0:
        return ['xue_xi_wen_ti','学习能力问题']
    if res == 1:
        return ['xue_xi_wen_ti','学习方法问题']
    if res == 2:
        return ['xue_xi_wen_ti','学习态度问题']
    if res == 3:
        return ['xue_xi_wen_ti','注意力问题']
    if res == 4:
        return []

def test_two_word_simi(text1, text2):

    bc = BertClient()
    test_vec1 = bc.encode([text1])
    test_vec2 = bc.encode([text2])
    tf.reset_default_graph()
    similary_value = calSimilarityByLearningModelPro(test_vec1, test_vec2)
    print(text1, text2, similary_value)



def calSimilarityByLearningModelPro(train_data, test_vec, topK=5):
    learning_rate = 0.001
    batch_size = 1# Network Parameters
    n_input = 1536  # Number of feature
    n_hidden_1 = 32  # 1st layer number of features
    n_classes = 2  # Number of classes to predict
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    def multilayer_perceptron(x, weights, biases):# Create model
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        model_file = tf.train.latest_checkpoint('ckpt_learning/')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        testbatch = [[] for i in range(1)]
        testbatch[0] = np.append(train_data, test_vec).tolist()
        res = sess.run(pred, feed_dict={x: testbatch})[0][0]
        print('################################',res)

    return res


def test(sentence):
    cls_tag = manage(sentence)
    print(cls_tag)


# sample to use
if __name__ == '__main__':
    l1 = '学习能力问题,理解能力差,学习能力差,学习障碍,记忆力差,记不住东西,忘性大,记得慢,阅读障碍,听写障碍,智力低下,上课听不懂,补课效果不好,无法理解,不能理解,学习有障碍,理解有难度,很难理解,总是理解不了,总是学不会,理解能力差,智商不高,听不懂课,理解不了上课内容,看不懂题目'
    l2 = '学习方法问题,学习很努力但是成绩不提高,不读英语,不会整理,学习没有计划,学习方法不当,严重偏科,抓不住重点,学习习惯差,边学习边玩,钻牛角尖,没有学习计划,不会规划学习时间,学习效率低,不会管理学习时间,没有掌握学习方法,不开窍,方法不当,没有掌握学习技巧,死记硬背'
    l3 = '学习态度问题,不想上学,想退学,不想上课,不爱学习,厌学,没有兴趣,无兴趣,缺乏兴趣,态度不端正,不完成作业,不按时交作业,不交作业,不做课后作业,拖拉作业,潦草,乱写乱画,不认真,上课不带课本,抄袭作业,不背书,抄作业,忘带作业,忘交作业,上课不回答问题,对成绩无所谓,对学习无所谓,放任自流,不想学习,讨厌学习,没有学习兴趣,不写作业,作业马虎,随意完成作业,上课似听非听,作业爱做不做,上课一个字也听不进去,交白卷,不参加考试'
    l4 = '注意力问题,注意力不集中,不集中,注意力保持时间短,分心,犯粗心错误,马虎,多动,出神,写错别字,注意力,无法专注做事,走神,发呆,开小差,注意力分散,注意力不持久'
    l5 = '打,伤人,打架,骂人,言语威胁,吓唬,吵架,诽谤,挑衅老师,敌视,漠视,上课小动作,捣乱打闹,教室乱跑,' \
         '扰乱课堂秩序,扰乱考场纪律,无视校规校纪,撒谎,抢劫' \
         '不爱说话,胆小怕事,自卑,胆战心惊,矛盾,情绪低落,抑郁,悲观,易怒,偏激,焦虑,情绪不稳定,' \
         '放纵,任性,沉迷游戏,早恋,自伤自残,失恋,哈哈哈,不知道,你好'
    s1 = l1.split(',')
    s2 = l2.split(',')
    s3 = l3.split(',')
    s4 = l4.split(',')
    s5 = l5.split(',')

    for sentence in s3:
        test(sentence)
    while(True):
        s = input('Input:')
        test(s)


