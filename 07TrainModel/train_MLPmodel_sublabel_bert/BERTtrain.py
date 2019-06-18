#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import numpy as np
from matplotlib import pylab
from ProcessData import process_data
import jieba
from bert_serving.client import BertClient
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split



def peredata(select):
    '''
    tf实现，不可以画图， 仅可以获得一个accuracy
    将句子通过BERT embedding转为vector，切分训练集和验证集
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    train_data = []  # 用于存储训练文档信息
    train_label = []  # 用于存储训练文档信息
    print('step 1 获取', select, '语句')
    texts, labels, classes_num = process_data(select)
    bc = BertClient()
    for i in range(len(texts)):
        v1 = bc.encode(texts[i])
        train_data.append(v1[0])
    train_label = np.array(labels)

    # indices = np.arange(len(train_data))  # shuffle
    # np.random.shuffle(indices)
    # train_data = train_data[indices]
    # train_label = train_label[indices]

    if not os.path.exists(select + "_train_data.npy"):  #save data
        np.save(select + "_train_data.npy", train_data)
    if not os.path.exists(select + "_train_label.npy"):
        np.save(select + "_train_label.npy", train_label)

    print('step2: BERT转换向量完成，存储在_train_data.npy, _train_label.npy')
    return train_data, train_label, classes_num


def GetTrainData(select):
    train_data = np.load(select + "_train_data.npy")
    train_label = np.load(select + "_train_label.npy")
    print('直接导入分类数据成功', select)
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)
    indices = np.arange(train_data.shape[0])  # shuffle
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]

    set_label = set(train_label)
    classes_num = len(set_label)
    if select =='all':
        classes_num = 90
    elif select == 'attack':
        classes_num = 3
    elif select == 'disorder':
        classes_num = 4
    else:
        classes_num = 3
    return train_data, train_label, classes_num


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
    training_epochs = 200

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
            # saver.save(sess, 'ckptann/mlp.ckpt', global_step=epoch)
            if epoch % display_step == 0:  # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))  # Accuracy: 0.9905


if __name__ == '__main__':
    print('采用BERT+MLP模型对标注文本做分类')
    # option = ['all', 'attack', 'disorder', 'pinxingwenti']
    option = ['pinxingwenti']
    for i in range(0, len(option)):
        select = option[i]
        if not os.path.exists(select + "_train_data.npy"):
            train_data, train_label, classes_num = peredata(select)
        else:
            train_data, train_label, classes_num = GetTrainData(select)  # 获取训练数据
        train_model(train_data, train_label, classes_num)
    # select = option[0]
    # train_data, train_label = GetTrainData(select)  # 获取训练数据
    # train_model(train_data, train_label)