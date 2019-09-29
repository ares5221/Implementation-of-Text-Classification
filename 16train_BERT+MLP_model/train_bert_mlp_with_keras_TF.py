#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import os
import time

from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

def peredata(index):
    '''
    keras实现，可以画图
    将句子通过BERT embedding转为vector
    :return:x_train, y_train, x_val, y_val即训练集和验证集的label和text
    '''
    train_data = []  # 用于存储训练文档信息
    ann_filename = './data/'+str(index) + '.csv'
    contents, tags = [], []
    with open(ann_filename, 'r', encoding='utf-8') as fcsv:
        csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:
            contents.append(row[1])
            tags.append(int(row[0]))

    classes_num = max(tags) +1 #类别个数

    bc = BertClient()
    for i in range(len(contents)):
        if True:
            v1 = bc.encode([contents[i]])
            train_data.append(v1[0])
    train_label = np.array(tags)

    # shuffle
    train_data = np.array(train_data)
    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]

    # save data in npy
    np.save('./data/' + str(index) + "_train_data.npy", train_data)
    np.save('./data/' +  str(index) + "_train_label.npy", train_label)
    print('标注语句通过BERT转换向量ok..')
    return train_data, train_label, classes_num


def GetTrainData(index):
    train_data = np.load('./data/' + str(index) + "_train_data.npy")
    train_label = np.load('./data/' +  str(index) + "_train_label.npy")
    print(index, '直接导入分类数据成功')
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)

    indices = np.arange(train_data.shape[0])  # shuffle
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]

    set_label = set(train_label)  #统计分类类别
    classes_num = len(set_label)
    print('数据共分为几类的情况：', set_label, classes_num)

    count = [0 for i in range(classes_num)]
    for label in train_label:
        for count_index in range(classes_num):
            if label == count_index:
                count[count_index] +=1
    print('第 ',index,' 个维度共有 ', len(train_label), '条数据，其中类别比例为 ', count)
    return train_data, train_label, classes_num


def train_model_keras(train_data, train_label, classes_num, index):
    if index == 20:
        classes_num = 14
    print('start build MLP model with keras...')
    max_len = 768
    epochs_num = 1000
    batch_size_num = 64

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            padding='post',
                                                            maxlen=max_len)
    train_label = np.array(train_label)
    train_label = np_utils.to_categorical(train_label, classes_num)

    #bulid model
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(768,)))
    model.add(keras.layers.Dense(classes_num, activation=tf.nn.softmax))
    model.summary()
    # 损失函数和优化
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_test_split_num = int(len(train_label) * 0.8)
    train_val_split_num = int(train_test_split_num * 0.8)
    x_val = train_data[train_val_split_num:train_test_split_num]
    partial_x_train = train_data[0:train_val_split_num]
    y_val = train_label[train_val_split_num:train_test_split_num]
    partial_y_train = train_label[0:train_val_split_num]
    test_data = train_data[train_test_split_num:]
    test_labels = train_label[train_test_split_num:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs_num,
                        batch_size=batch_size_num,
                        validation_data=(x_val, y_val),
                        shuffle=True)

    results = model.evaluate(test_data, test_labels)
    print(index,' -->keras model评估模型效果(损失-精度）：...', results)

    # print('step6: 开始绘图...')
    # history_dict = history.history
    # history_dict.keys()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)
    # # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # tt1 = str(index) + ' :Training and validation loss'
    # plt.title(tt1)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # plt.clf()  # clear figure
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # tt2 = str(index) + ' :Training and validation accuracy'
    # plt.title(tt2)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    # print('模型训练结束！！！！！')
    return results[1]



def train_model_tf(train_data, train_label, classes_num, index):
    print('start build MLP model with tensorflow...')
    if index == 20:
        classes_num = 14
    train_label = np_utils.to_categorical(train_label, classes_num)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.4)
    start_time = time.time()
    tf.reset_default_graph()
    learning_rate = 0.001
    training_epochs = 300
    batch_size = 64
    display_step = 10

    # Network Parameters
    n_input = 768  # Number of feature
    n_hidden_1 = 64  # 1st layer number of features
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
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                avg_cost += c / total_batch  # Compute average loss
            # saver.save(sess, 'ckptann/mlp.ckpt', global_step=epoch)
            if epoch % display_step == 0:  # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(index, "-->TF model Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
        return accuracy.eval({x: X_test, y: Y_test})


if __name__ == '__main__':
    print('采用BERT+MLP模型对标注文本做分类...')
    option = [i for i in range(1,21)]#数据文件名索引为1-20
    for index in option:
        if index ==17 or index == 15:
            continue
        if True:
        # if index==15:
            if os.path.exists('./data/' +  str(index) + "_train_data.npy") and os.path.exists('./data/' +  str(index) + "_train_label.npy"):
                train_data, train_label, classes_num = GetTrainData(index)  # 获取训练数据
            else:
                train_data, train_label, classes_num = peredata(index)
            keras_sum = 0
            for i in range(10):
                #train in keras-Impl
                keras_acc = train_model_keras(train_data, train_label, classes_num, index)
                keras_sum +=keras_acc
            with open("keras_acc.csv", "a", encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([index, keras_sum/10.0])

            tf_sum = 0
            for i in range(10):
                # train in tf-Impl
                tf_acc = train_model_tf(train_data, train_label, classes_num, index)
                tf_sum += tf_acc
            with open("tf_acc.csv", "a", encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([index, tf_sum / 10.0])
