#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json
import os
import jieba
import tensorflow as tf
from tensorflow import keras

def get_predict_data():
    db_dir = r'G:\tf-start\Implementation-of-Text-Classification\dataset'
    filename = 'work'  # 需要分类处理的文档路径
    work_dir = os.path.join(db_dir, filename)

    texts = []
    for fname in os.listdir(work_dir):
        if fname[-4:] == '.txt':
            f = open(os.path.join(work_dir, fname), 'r', encoding='UTF-8')
            texts.append(f.read())
            f.close()
    return texts


def pereworkdata(data):
    work_data = []
    seg_content_list = [[] for index in range(len(data))]  # 存储每条content的分词结果
    count_word_frequency = {}  # 存储文档中出现的词
    with open('word_index_dict.json', 'r', encoding='utf-8') as load_f:
        word_index_dic = json.load(load_f)
        print(word_index_dic)
    for i in range(len(data)):
        word_list = []
        seg_content_data = jieba.cut(data[i])
        for word in seg_content_data:
            if word in word_index_dic:
                word_list.append(word_index_dic[word])
            else:
                word_list.append(0)
        work_data.append(word_list)
    print(work_data)
    return work_data



if __name__ == '__main__':
    worktexts = get_predict_data()
    data = pereworkdata(worktexts)
    train_data = keras.preprocessing.sequence.pad_sequences(data,
                                                            padding='post',
                                                            maxlen=300)
    print('Shape of data tensor:', train_data.shape)
    model = keras.models.load_model('pre_trained_model_1.h5')

    test_acc = model.predict(train_data)
    print('评估模型效果(损失-精度）：...', test_acc)