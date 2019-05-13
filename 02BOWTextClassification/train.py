#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import numpy as np
from matplotlib import pylab
from ProcessData import process_data
import jieba
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# settings
training_samples = 300
validation_samples = 264
max_words = 28918 + 2

'''采用词袋模型来编码文章'''


def peredata():
    '''
    将texts分词，向量化，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    # step 1 获取所有词汇
    filename = 'train'
    labels, texts = process_data(filename)
    train_label = np.array(labels)
    np.save("train_label.npy", train_label)  # 保存所有标签

    word_set = set()  # 存储文档中全部出现的词--词表
    word_dic = []  # 去除停用词后的词表
    seg_content_list = [[] for index in range(len(texts))]  # 存储每条content的分词结果
    for i in range(len(texts)):
        seg_content_data = jieba.cut(texts[i])
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_set:
                word_set.add(word)
    # print('获取的全部词表', word_set)
    if not os.path.exists('word_set.txt'):
        with open('word_set.txt', 'w', encoding='utf-8') as f:
            f.write(str(word_set))
        print('统计文档中词的出现个数', len(word_set))  # 28918
    else:
        print('该文件已经存在，不再创建。。。')
        print('统计文档中词的出现个数', len(word_set))

    # step2 去除停用词及其他无效字符等
    stopword_list = []
    for line in open("stopword.txt", "r", encoding='utf-8'):  # 设置文件对象并读取每一行文件
        stopword_list.append(line.strip('\n'))
    # print('当前停用词表内容：', stopword_list)
    if not os.path.exists('word_dic.txt'):
        for wd in word_set:
            if wd not in stopword_list:
                word_dic.append(wd)
            with open('word_dic.txt', 'w', encoding='utf-8') as f:
                f.write(str(word_dic))
    else:
        word_dic = []
        f_word_dic = open("word_dic.txt", "r", encoding='utf-8')  # 设置文件对象
        str_word = f_word_dic.read()  # 将txt文件的所有内容读入到字符串str_word中
        str_word = str_word[1:-1]  # 字符串处理
        for sw in str_word.split(','):
            sw = sw.lstrip().lstrip('\'').rstrip('\'')
            word_dic.append(sw)
        f_word_dic.close()  # 将文件关闭
        print('读取到词表的大小为： ', len(word_dic))
        print(word_dic)
    print('通过停用词表清除词汇后词表个数：', len(word_dic))
    # for sss in seg_content_list:
    #     print(sss)
    # print(seg_content_list)
    # step3 TF-IDF
    # vectorizer = CountVectorizer(min_df=1e-5)  # drop df < 1e-5,去低频词
    # transformer = TfidfTransformer()
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_set))
    # words = vectorizer.get_feature_names()

    print(word_dic)
    return seg_content_list, word_dic, labels


# step 3  对seg_content_list 用词袋编码(如果只置为1是one-hot编码)
def EncodeByWordBag1(text_list, word_dic):
    text_num = len(text_list)
    word_len = len(word_dic)
    train_data = [[] for i in range(text_num)]  # 用于存储训练文档信息
    for i in range(text_num):
        word_list = [0 for x in range(word_len)]
        for j in range(len(text_list[i])):
            if text_list[i][j] in word_dic:
                idx = word_dic.index(text_list[i][j])
                word_list[idx] += 1
            # else:
            #     word_list[idx] = 0
        print(len(word_list), sum(word_list), word_list)
        train_data[i] = word_list
    np.save("train_data.npy", train_data)  # 保存编码后的数据
    return train_data


# step 3  统计每个词出现的次数然后除以文章词的总数，将比值作为编码值（词袋模型归一化）
def EncodeByWordBag2(text_list, word_dic):
    text_num = len(text_list)
    word_len = len(word_dic)
    train_data = [[] for i in range(text_num)]  # 用于存储训练文档信息
    for i in range(text_num):
        word_list = [0 for x in range(word_len)]
        for j in range(len(text_list[i])):
            if text_list[i][j] in word_dic:
                idx = word_dic.index(text_list[i][j])
                word_list[idx] += 1
        # print('encode2', len(word_list), sum(word_list), word_list)
        # print(word_list)
        word_list = [kk / (len(text_list[i])) for kk in word_list]
        print(i, len(word_list), word_list, len(text_list[i]))
        train_data[i] = word_list
    np.save("train_data2.npy", train_data)  # 保存编码后的数据
    return train_data


def GetTrainData():
    train_data = np.load("train_data2.npy")
    # print(train_data)
    train_label = np.load("train_label.npy")
    # print(train_label)
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    # shuffle
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]
    print(train_label)
    return train_data, train_label


def train_model(train_data, train_label):
    '''
    训练模型
    :return:训练时loss,acc
    '''
    x_train = train_data[:training_samples]
    y_train = train_label[:training_samples]
    x_val = train_data[training_samples: training_samples + validation_samples]
    y_val = train_label[training_samples: training_samples + validation_samples]
    start_time = time.time()
    # # LogisticRegression classiy model
    average = 0
    testNum = 50
    for i in range(0, testNum):
        # x_train, x_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.4)
        lr_model = LogisticRegression(solver='liblinear', C=1e6, penalty='l2')
        lr_model.fit(x_train, y_train)
        print("val mean accuracy: {0}".format(lr_model.score(x_val, y_val)))
        y_pred = lr_model.predict(x_val)
        p = np.mean(y_pred == y_val)
        print(p)
        average += p
    # precision and recall
    answer = lr_model.predict_proba(x_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, answer)
    report = answer > 0.5
    print(classification_report(y_val, report, target_names=['neg', 'pos']))
    print("average precision:", average / testNum)
    print("time spent:", time.time() - start_time)
    plot_pr(0.5, precision, recall, "pos")

    # 保存Model(注:save文件夹要预先建立，否则会报错)
    with open('LRModel--1.pickle', 'wb') as f:
        pickle.dump(lr_model, f)
    # # SVM classiy model
    # svm_model = SVC(gamma='auto')
    # svm_model.fit(x_train, y_train)
    # print("val mean accuracy: {0}".format(svm_model.score(x_val, y_val)))
    # # 保存SVM Model
    # with open('SVMModel.pickle', 'wb') as f:
    #     pickle.dump(svm_model, f)


# Draw R/P Curve
def plot_pr(auc_score, precision, recall, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    pylab.show()


if __name__ == '__main__':
    if 1 == 0:
        seg_texts, word_dic, labels = peredata()  # 获取全部的词表，及文章分词情况
        # train_data = EncodeByWordBag1(seg_texts, word_dic)  # 编码
        train_data = EncodeByWordBag2(seg_texts, word_dic)

    train_data, train_label = GetTrainData()  # 获取训练数据
    train_model(train_data, train_label)
