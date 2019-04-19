#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import timeit
import numpy as np
import json
import xlrd, xlwt
from tensorflow import keras
import matplotlib.pyplot as plt
# import jieba
import pkuseg
'''实现文本数据针对攻击行为的三分类及违纪行为的四分类
更新分词方式
'''
filePath = r'G:\tf-start\tensorflow-learning-nlp\04ai-teacher\data\demodata0103.xlsx'
train_data = []  # 用于存储训练文档信息
train_label = []  # 用于存储excel中第三列是否攻击的label数据（特征）
train_label1 = []  # 用于存储excel中第四列是否违纪的label数据（特征）

# 由于这里数据量较少，测试数据也是从train_data, train_label中截取出来的
# test_data = []
# test_label = []



def savedata(title):
    print('step2：通过title获取result.json中的content...')
    rj_list = loadResultJson()
    train_content_data = ['' for index in range(len(title))]  # 用于存储result.json中title对应的content
    for i in range(len(title)):
        for j in range(len(rj_list)):
            if rj_list[j]['title'] == title[i]:
                train_content_data[i] = rj_list[j]['content']
    return train_content_data

def peredata(content_data):
    print('step3: 分词处理及数据预处理...')
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    seg_content_list = [[] for index in range(len(content_data))]  # 存储每条content的分词结果
    seg = pkuseg.pkuseg()
    for i in range(len(content_data)):
        seg_content_data = seg.cut(content_data[i])
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_index_dic:
                word_index_dic[word] = len(word_index_dic) + 1

    max_words = len(word_index_dic)  # 统计得到该文档用到的词的个数16895
    with open('word_index_dict.json', 'w', encoding='utf-8') as f:
        json.dump(word_index_dic, f, ensure_ascii=False)  # json.dumps在默认情况下，对于非ascii字符生成的是相对应的字符编码，而非原始字符
    # print('文档中出现的词已经全部统计编码完，新的分词方式分词后得到16895个词。')

    for j in range(len(seg_content_list)):
        word_list = []
        for word in seg_content_list[j]:
            word_list.append(word_index_dic[word])
        train_data.append(word_list)
    save_traindata_txt(train_data)
    return train_data


def save_traindata_txt(train_data):
    print('将traindata保存在txt文件中')
    file = open('train_data.txt', 'w')
    for datalist in train_data:
        # print(datalist, 'xxxxx')
        file.write(str(datalist))
        file.write('\n')
    file.close()


def get_max_content_length(data_list):
    max_len, index = 0, 0
    for i in range(len(data_list)):
        # print(data_list[i])
        if max_len < len(data_list[i]):
            max_len = len(data_list[i])
            index = i
    print('content中最长的文本长度为：', max_len, 'index：', index)
    return max_len


@clock
def build_model(train_data,train_label):
    print('step4: start build model...')
    max_len = get_max_content_length(train_data)
    print(max_len)
    # max_len = 3000  # 计算得到文本最大长度为3496
    max_words = 17000  # 统计得到该文档用到的词的个数19307/20000
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)

    train_label = np.array(train_label)
    model = keras.Sequential()
    model.add(keras.layers.Embedding(max_words, 16))
    model.add(keras.layers.LSTM(128, activation=tf.nn.tanh))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(4, activation=tf.nn.softmax))
    # model.add(keras.layers.Activation(tf.nn.softmax))
    model.summary()
    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[0:24]
    partial_x_train = train_data[0:]
    y_val = train_label[0:24]
    partial_y_train = train_label1[0:]

    test_data = train_data[0:20]
    test_labels = train_label[0:20]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=32,
                        validation_data=(x_val, y_val)
                        )

    results = model.evaluate(test_data, test_labels)
    print('step5: 评估模型效果(损失-精度）：...', results)

    print('step6: 开始绘图...')
    history_dict = history.history
    print(history.history)
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()  # clear figure
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print('模型训练结束！！！！！')


def loadResultJson():
    # dumps和loads是在内存中转换（python对象和json字符串之间的转换），而dump和load则是对应于文件的处理。
    resultpath = r'G:\tf-start\tensorflow-learning-nlp\04ai-teacher\data\result.json'
    f = open(resultpath, encoding='utf-8')
    tt = json.load(f)  # 通过load获取result.json中list格式的data
    return tt


def execute():
    title_list = read_excel()
    content_data = savedata(title_list)
    x_data = peredata(content_data)
    build_model(x_data, train_label)


if __name__ == '__main__':
    execute()
