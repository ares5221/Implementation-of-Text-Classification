#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import os
from bert_serving.client import BertClient
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        model_file = tf.train.latest_checkpoint('ckpt_learning_sentences/')
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



def train_knnmodel(train_data, train_label, sentence, topK):
    '''调用训练好的攻击行为 关键词的mlp模型,找到其中所属最多的label'''
    # topK = 10  # K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    tf.reset_default_graph()
    idlist, simlist =calSimilarityByLearningModel(train_data, test_vec, topK)
    # 根据找到的最相似关键词的索引 获取其对应的label信息,判断其具体属于那个二级类别如身体攻击还是言语攻击
    # print('通过MLP计算 输入语句( %s )与所有关键词的相似度前%d个:'%(sentence,topK), simlist)
    # print('最相似词的对应索引index:', idlist)
    cal_lab = []
    for idx in idlist:
        cal_lab.append(train_label[idx])
    sent_predict = max(cal_lab, key=cal_lab.count)
    return sent_predict


def KNN_learning(sentence, topK):
    if os.path.exists(DIR + "learn_sentences_387.npy") and os.path.exists(DIR + "learn_labels_387.npy"):
        train_data = np.load(DIR + "learn_sentences_387.npy")
        train_label = np.load(DIR + "learn_labels_387.npy")
    else:
        pass # 这里会调用生成attack_words_77.npy的方法
    res = train_knnmodel(train_data, train_label, sentence, topK)
    if res == 0:
        return ['xue_xi_wen_ti','学习能力问题']
    if res == 1:
        return ['xue_xi_wen_ti','学习方法问题']
    if res == 2:
        return ['xue_xi_wen_ti','学习态度问题']
    if res == 3:
        return ['xue_xi_wen_ti','注意力问题']



def test(sentence):
    cls_tag = manage(sentence)
    print(cls_tag)


def test_MLP_KNN_Acc(topK):
    #read test file
    testCSV = DIR + 'testKNN_learn_sentences_labels.csv'
    test_sentences, test_labels = [], []
    with open(testCSV, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            test_sentences.append(i[0])
            test_labels.append(i[1])
    # print(test_sentences)
    # print(test_labels)
    # test model acc
    correct_count = 0

    for index in range(len(test_sentences)):
        test_res = KNN_learning(test_sentences[index],topK)
        if '能力' in str(test_res):
            test_label = 0
        elif '方法' in str(test_res):
            test_label = 1
        elif '态度' in str(test_res):
            test_label = 2
        else:
            test_label = 3
        # print('当前测试句子根据模型得到的标签为 ', test_label, '实际标签是', test_labels[index])
        if test_label == int(test_labels[index]):
            correct_count += 1
    acc = correct_count/len(test_labels)
    print('K setting', topK, 'MLP+KNN模型的精度=', acc)
    return acc

# sample to use
if __name__ == '__main__':
    # for sentence in ['英语学科上还偏科', '学习缺乏兴趣，没有主动性和自律性。学习基础不扎实，经常迟到，几乎完不成作业',
    #                  '学习成绩一团糟，初二的学生还不知道“分数”为何物，“方程”就更别说了。',
    #                  '学习兴趣不高，不是特别在意成绩']:
    #     test(sentence)

    #计算MLP+KNN模型的准确度
    K, accs = [], []
    for topK in range(1,21):
        acc = test_MLP_KNN_Acc(topK)
        accs.append(acc)
        K.append(topK)
    # Drow Pic
    plt.plot(K, accs, marker='*', mec='r', mfc='w')
    plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"K setting")  # X轴标签
    plt.ylabel("Accurcy")  # Y轴标签
    # plt.title("A simple plot")  # 标题
    plt.show()

    while(True):
        s = input('Input:')
        test(s)

'''
result output:
['xue_xi_wen_ti', '学习方法问题']
['xue_xi_wen_ti', '学习态度问题']
['xue_xi_wen_ti', '学习方法问题']
['xue_xi_wen_ti', '学习态度问题']
'''

