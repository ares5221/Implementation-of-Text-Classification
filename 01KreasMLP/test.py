#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json
import os
import jieba
import tensorflow as tf
from tensorflow import keras
from GetWorkFiles import Translate

db_dir = r'G:\tf-start\Implementation-of-Text-Classification\dataset'
filename = r'work\bzrlt'  # 需要分类处理的文档路径
# filename = r'work\dyal'  # 需要分类处理的文档路径
# filename = r'work\wenku'  # 需要分类处理的文档路径
# filename = r'work\wenku_deyuanli'  # 需要分类处理的文档路径
work_dir = os.path.join(db_dir, filename)
# settings
max_len = 200

def get_predict_data():
    texts = []
    for fname in os.listdir(work_dir):
        if fname[-4:] == '.txt':
            f = open(os.path.join(work_dir, fname), 'r', encoding='UTF-8')
            tmp = f.read()
            if len(tmp) < 30:
                print(fname)
            tmp = tmp[20:]
            # print('#############',tmp)
            texts.append(tmp)
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
    # for ll in worktexts:
    #     print(len(ll))

    data = pereworkdata(worktexts)
    train_data = keras.preprocessing.sequence.pad_sequences(data,
                                                            padding='post',
                                                            maxlen=max_len)
    print('Shape of data tensor:', train_data.shape)
    # 导入模型
    # model = keras.models.load_model('pre_trained_MLPmodel_1.h5')
    model = keras.models.load_model('pre_trained_CNNmodel_1.h5')
    # model = keras.models.load_model('pre_trained_LSTMmodel_1.h5')
    test_acc = model.predict(train_data)
    for i in range(len(test_acc)):
        # print(test_acc[i])
        if test_acc[i] > 0.6:
            print('文件：', i, '...', test_acc[i])
            # Translate(work_dir, i)

    print('over')
