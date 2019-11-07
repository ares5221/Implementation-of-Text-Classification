#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from bert_serving.client import BertClient
import numpy as np
import tensorflow as tf
import json
# DIR = 'semi/nlp_model/'
DIR = './data/'

def manage(sentence):
    if len(sentence) >0:
        classification_tag = KNN_attack(sentence)
    else:
        print('invalid sentence input!!')
        return
    return classification_tag


def calSimilarityByAttackModel(train_data, test_vec, topK=5):
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
        out_layer = tf.nn.softmax(out_layer)
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
        model_file = tf.train.latest_checkpoint('ckpt_attack/')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        simlist, idlist = [0 for i in range(topK)], [0 for i in range(topK)]
        for i in range(len(train_data)):
            testbatch = [[] for i in range(1)]
            testbatch[0] = np.append(train_data[i], test_vec).tolist()
            res = sess.run(pred, feed_dict={x: testbatch})[0][0]
            # print(res, i)
            if res > min(simlist):
                simlist[simlist.index(min(simlist))] = res
                idlist[simlist.index(min(simlist))] = i
    return idlist, simlist



def train_knnmodel(train_data, train_label, sentence):
    '''调用训练好的攻击行为 关键词的mlp模型,找到其中所属最多的label'''
    topK = 5 # K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    tf.reset_default_graph()
    idlist, simlist =calSimilarityByAttackModel(train_data, test_vec, topK)
    # 根据找到的最相似关键词的索引 获取其对应的label信息,判断其具体属于那个二级类别如身体攻击还是言语攻击
    print('通过MLP计算 输入语句( %s )与所有关键词的相似度前topK个:'%sentence, simlist)
    # print('最相似词的对应索引index:', idlist)
    get_real_word(idlist)
    cal_lab = []
    for idx in idlist:
        cal_lab.append(train_label[idx])
    sent_predict = max(cal_lab, key=cal_lab.count)
    return sent_predict


def get_real_word(most_index):
    index_2_keyword_name = './data/attack_words_index_2_keyword.json'
    ss = []
    with open(index_2_keyword_name, 'r', encoding='utf-8') as f:
        word_index_dic = json.load(f)
        for index in most_index:
            ss.append(word_index_dic[str(index)])
    print('最相似的词是：', ss)





def KNN_attack(sentence):
    if os.path.exists(DIR + "attack_words_110.npy") and os.path.exists(DIR + "label_attack_words_114.npy"):
        train_data = np.load(DIR + "attack_words_110.npy")
        train_label = np.load(DIR + "label_attack_words_114.npy")
    else:
        pass # 这里会调用生成attack_words_77.npy的方法
    res = train_knnmodel(train_data, train_label, sentence)
    if res == 0:
        return ['gong_ji_xing_wei','身体攻击行为']
    if res == 1:
        return ['gong_ji_xing_wei','言语攻击行为']
    if res == 2:
        return ['gong_ji_xing_wei','关系攻击行为']
    if res == 3:
        return []



def test(sentence):
    cls_tag = manage(sentence)
    print(cls_tag)


# sample to use
if __name__ == '__main__':

    a1 = '身体攻击行为,打,伤人,打架,斗殴,暴力,好斗,踢打,伤害,踹,蹬,挠,怼,砸,掐,扎,推,骚扰,欺负,霸道,吐口水,朝同学泼开水,用粉笔头丢同学,破坏同学东西,撞同学,向同学扔东西,扇耳光,划伤,身体冲撞,殴打,打群架,推搡同学'
    a2 = '言语攻击行为,嘲笑,骂人,言语威胁,吓唬,吵架,言语冲撞,顶撞,吼同学,言语恐吓,言语挑衅,揭短,言语凌辱,顶嘴,责骂,取笑,说脏话,抬杠,与父母大吵大闹,辱骂,起外号,嘲讽'
    a3 = '关系攻击行为,诽谤,挑衅老师,敌视,漠视,回避,敌意,与老师对着干,摆脸色,故意不搭理,不听话,诋毁,对着干,说别人坏话,捉弄同学,污蔑,讽刺,诋毁,挑拔,离间,孤立同学'
    a4 = '不听讲,看课外书,摆弄东西,学习障碍,扔纸团,扰乱考场纪律,不遵守规章制度,无视校规校纪,撒谎,' \
         '沉默寡言,不爱说话,胆小怕事,自卑,胆战心惊,矛盾,情绪低落,抑郁,悲观,焦虑,情绪不稳定,理解能力差,学习能力差,' \
         '缺乏兴趣,态度不端正,放纵,任性,多动,注意力,沉迷游戏,早恋,自伤自残,失恋,哈哈哈,不知道,你好'
    s1 = a1.split(',')
    s2 = a2.split(',')
    s3 = a3.split(',')
    s4 = a4.split(',')

    for sentence in s3:
        test(sentence)
    while(True):
        s = input('Input:')
        test(s)

