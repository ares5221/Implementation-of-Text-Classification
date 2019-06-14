#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from ProcessData import process_data
import jieba
import time
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split



def peredata(select):
    '''
    将texts分词，向量化，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    # step 1 获取所有词汇
    texts, labels, classes_num = process_data(select)
    train_label = np.array(labels)
    if not os.path.exists(select + "_train_label.npy"):
        np.save(select + "_train_label.npy", train_label)  # 保存所有标签

    word_set = set()  # 存储文档中全部出现的词--词表
    word_dic = []  # 去除停用词后的词表
    seg_content_list = [[] for index in range(len(texts))]  # 存储每条content的分词结果
    for i in range(len(texts)):
        seg_content_data = jieba.cut(texts[i][0])
        for word in seg_content_data:
            seg_content_list[i].append(word)
            if word not in word_set:
                word_set.add(word)
    word_set_name = select + '_word_set.txt'
    if not os.path.exists(word_set_name):
        with open(word_set_name, 'w', encoding='utf-8') as f:
            f.write(str(word_set))
        print('统计文档中词的出现个数', len(word_set))
    else:
        print('该文件已经存在，不再创建。。。', select)
        print('统计文档中词的出现个数', len(word_set))

    # step2 去除停用词及其他无效字符等
    stopword_list = []
    for line in open("stopword.txt", "r", encoding='utf-8'):  # 设置文件对象并读取每一行文件
        stopword_list.append(line.strip('\n'))
    # print('当前停用词表内容：', stopword_list)
    word_dic_name = select + '_word_dic.txt'
    if not os.path.exists(word_dic_name):
        for wd in word_set:
            if wd not in stopword_list:
                word_dic.append(wd)
            with open(word_dic_name, 'w', encoding='utf-8') as f:
                f.write(str(word_dic))
    else:
        word_dic = []
        f_word_dic = open(word_dic_name, "r", encoding='utf-8')  # 设置文件对象
        str_word = f_word_dic.read()  # 将txt文件的所有内容读入到字符串str_word中
        str_word = str_word[1:-1]  # 字符串处理
        for sw in str_word.split(','):
            sw = sw.lstrip().lstrip('\'').rstrip('\'')
            word_dic.append(sw)
        f_word_dic.close()  # 将文件关闭
    print('通过停用词表清除词汇后词表个数：', len(word_dic), word_dic)
    return seg_content_list, word_dic, labels, classes_num


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
def EncodeByWordBag2(text_list, word_dic, select):
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
        # print(i, len(word_list), word_list, len(text_list[i]))
        train_data[i] = word_list
    np.save(select + "_train_data.npy", train_data)  # 保存编码后的数据
    return train_data


def GetTrainData(select):
    train_data = np.load(select + "_train_data.npy")
    train_label = np.load(select + "_train_label.npy")
    print('导入分类数据成功')
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    indices = np.arange(train_data.shape[0])  # shuffle
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]
    return train_data, train_label


def train_model(train_data, train_label, classes_num):
    '''
    训练模型
    :return:训练时loss,acc
    '''
    train_label = np_utils.to_categorical(train_label, classes_num)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.3)
    start_time = time.time()
    tf.reset_default_graph()
    learning_rate = 0.001
    training_epochs = 400
    batch_size = 64
    display_step = 100

    # Network Parameters
    n_input = len(train_data[0])  # Number of feature
    n_hidden_1 = 32  # 1st layer number of features
    n_classes = classes_num  # Number of classes to predict

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Create model
    def multilayer_perceptron(x, weights, biases):
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

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=4)  # save model
        for epoch in range(training_epochs):  # Training cycle
            avg_cost = 0.
            if len(X_train) / batch_size <1:
                total_batch = 1
            else:
                total_batch = int(len(X_train) / batch_size)
            X_batches = np.array_split(X_train, total_batch)
            Y_batches = np.array_split(Y_train, total_batch)
            for i in range(total_batch):  # Loop over all batches
                batch_x, batch_y = X_batches[i], Y_batches[i]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                avg_cost += c / total_batch  # Compute average loss
            saver.save(sess, 'ckptann/mlp.ckpt', global_step=epoch)
            if epoch % display_step == 0:  # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))  # Accuracy: 0.9905


if __name__ == '__main__':
    print('采用词袋模型来编码标注文本做分类')
    # option = ['all', 'attack', 'disorder', 'pinxingwenti']
    option = ['pinxingwenti']
    for i in range(0, len(option)):
        select = option[i]
        seg_texts, word_dic, labels, classes_num = peredata(select)  # 获取全部的词表，及文章分词情况
        train_data = EncodeByWordBag2(seg_texts, word_dic, select)  # 编码
        train_data, train_label = GetTrainData(select)  # 获取训练数据
        train_model(train_data, train_label, classes_num)
    # select = option[0]
    # train_data, train_label = GetTrainData(select)  # 获取训练数据
    # train_model(train_data, train_label)