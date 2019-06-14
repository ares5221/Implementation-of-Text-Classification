#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
from keras.utils import np_utils

'''
训练标注文本分类的MLP模型 X_data Y_data为label_text.csv通过bert embedding后的结果及对应标签
2824条标注数据 2019/6/11 updata   Accuracy: 0.25925925
'''

X = np.load('X_data.npy')
Y = np.load('Y_data.npy')
print('导入分类数据成功', X.shape, Y.shape)

Y_label = np_utils.to_categorical(Y, 95)
X_train, X_test = X[0:2500], X[2500:]
Y_train, Y_test = Y_label[0:2500], Y_label[2500:]

tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 400
batch_size = 64
display_step = 100

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
        saver.save(sess, 'ckptann/mlp.ckpt', global_step=epoch)
        if epoch % display_step == 0:  # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))  # Accuracy: 0.9905
    # global result
    # result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})
