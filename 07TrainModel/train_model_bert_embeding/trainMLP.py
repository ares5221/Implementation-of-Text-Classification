#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
from bert_serving.client import BertClient
import os

'''
训练问题-答案的MLP模型 通过数据集X_qa_all_data Y_qa_all_data
为AllquestAnsWroA.csv通过bert encode后的结果
2969*2=5938       4/9
3291*2=6582条数据 4/15 updata
'''

X = np.load('X_data.npy')
Y = np.load('Y_data.npy')
Y_label = [[0 for j in range(95)] for i in range(Y.shape[0])]
print('导入问题-答案训练数据成功', X.shape, Y.shape)

for i in range(Y.shape[0]):
    Y_label[i][Y[i]-1] = 1
print(Y_label)
Y_label = np.array(Y_label)
# print(X_train, Y_train)
# print(X_test.shape, Y_test.shape)
X_train, X_test = X[0:2000], X[2000:]
Y_train, Y_test = Y_label[0:2000], Y_label[2000:]

tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 800
batch_size = 100
display_step = 10

# Network Parameters
n_input = 768  # Number of feature
n_hidden_1 = 32  # 1st layer number of features
n_classes = 95  # Number of classes to predict

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
        saver.save(sess, 'ckptque2ans/mlp.ckpt', global_step=epoch)
        if epoch % display_step == 0:  # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))  # Accuracy: 0.9905
    # global result
    # result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})
