#!/usr/bin/env python
# _*_ coding:utf-8 _*_
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
    topK = 10  # K setting
    bc = BertClient()
    test_vec = bc.encode([sentence])
    tf.reset_default_graph()
    idlist, simlist =calSimilarityByLearningModel(train_data, test_vec, topK)
    # 根据找到的最相似关键词的索引 获取其对应的label信息,判断其具体属于那个二级类别如身体攻击还是言语攻击
    print('通过MLP计算 输入语句( %s )与所有关键词的相似度前%d个:'%(sentence,topK), simlist)
    print('最相似词的对应索引index:', idlist)
    cal_lab = []
    for idx in idlist:
        cal_lab.append(train_label[idx])
    sent_predict = max(cal_lab, key=cal_lab.count)
    return sent_predict


def KNN_learning(sentence):
    if os.path.exists(DIR + "learn_words_145.npy") and os.path.exists(DIR + "label_learn_words_145.npy"):
        train_data = np.load(DIR + "learn_words_145.npy")
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



def test(sentence):
    cls_tag = manage(sentence)
    print(cls_tag)


# sample to use
if __name__ == '__main__':
    for sentence in ['攻击', '孩子学习没有计划', '孩子不写作业', '孩子注意力不集中']:
        test(sentence)
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

