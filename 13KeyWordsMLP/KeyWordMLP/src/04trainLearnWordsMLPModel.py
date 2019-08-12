#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
from bert_serving.client import BertClient
import os

'''
训练学习问题关键词的MLP模型 通过数据
X_train_all_learn_words_10098  Y_train_all_learn_words_10098,
bert embedding后的结果
'''

X = np.load(os.path.abspath('.') + '/data/X_train_all_learn_words_10098.npy')
Y = np.load(os.path.abspath('.') + '/data/Y_train_all_learn_words_10098.npy')
Y_label = np.array([Y, -(Y - 1)]).T
print('导入问题-问题训练数据成功', X.shape, Y_label.shape)

print(X.shape, Y_label.shape)
X_train, X_test = X[0:9000], X[9000:]
Y_train, Y_test = Y_label[0:9000], Y_label[9000:]
print(X[0], Y[0])
print(X_train, Y_train)
print(X_test.shape, Y_test.shape)

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 10

# Network Parameters
n_input = 1536  # Number of feature
n_hidden_1 = 32  # 1st layer number of features
n_classes = 2  # Number of classes to predict

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
        saver.save(sess, 'ckpt_learning/mlp.ckpt', global_step=epoch)
        if epoch % display_step == 0:  # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))  # Accuracy: 0.8992
    # global result
    # result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})
