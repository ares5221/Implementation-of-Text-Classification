#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
from bert_serving.client import BertClient
import os

'''
训练学习问题标注数据的MLP模型 通过数据
X_train_all_learn_words_10098  Y_train_all_learn_words_10098,
bert embedding后的结果
'''

X_train = np.load(os.path.abspath('.') + '/data/X_train_learn_sentences_387.npy')
Y_train = np.load(os.path.abspath('.') + '/data/Y_train_learn_sentences_387.npy')
print('导入学习问题标注语句-分类标签 训练数据成功', X_train.shape, Y_train.shape)
X_test = np.load(os.path.abspath('.') + '/data/X_test_learn_sentences_50.npy')
Y_test = np.load(os.path.abspath('.') + '/data/Y_test_learn_sentences_50.npy')
print('导入学习问题标注语句-分类标签 测试数据成功', X_test.shape, Y_test.shape)
# 标签数据转one_hot向量
classes = max(Y_train) + 1 ##类别数为最大数加1
one_hot_label = np.zeros(shape=(Y_train.shape[0],classes))##生成全0矩阵
one_hot_label[np.arange(0,Y_train.shape[0]),Y_train] = 1##相应标签位置置1
Y_train = one_hot_label

one_hot_label = np.zeros(shape=(Y_test.shape[0],classes))##生成全0矩阵
one_hot_label[np.arange(0,Y_test.shape[0]),Y_test] = 1##相应标签位置置1
Y_test = one_hot_label

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 10

# Network Parameters
n_input = 768  # Number of feature
n_hidden_1 = 32  # 1st layer number of features
n_classes = 4  # Number of classes to predict

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
        total_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(Y_train, total_batch)
        for i in range(total_batch):  # Loop over all batches
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            avg_cost += c / total_batch  # Compute average loss
        saver.save(sess, 'ckpt_learning_sentences/mlp.ckpt', global_step=epoch)
        if epoch % display_step == 0:  # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))  # Accuracy: 0.8992
    # global result
    # result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})
