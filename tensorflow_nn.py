# -*- coding:utf-8 -*-
# author:w.s
# software: PyCharm

import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LIG_LEVEL"] = "2"
tf.set_random_seed(777)

TB_SUMMARY_DIR = "./logs"  # TensorBroad Save path

# hidden layer function
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


X_data = np.linspace(-1, 1, num=300)
X_data = np.c_[X_data]

noise = np.random.normal(0, 0.05, X_data.shape).astype(np.float32)  # Definition Noise

y_data = np.square(X_data) - 0.5 + noise

X = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(X, 1, 20, activation_function=tf.nn.relu)

prediction = add_layer(l1, 20, 1, activation_function=None)

# loss function
loss = tf.reduce_mean(tf.reduce_sum(
    tf.square(y - prediction), reduction_indices=[1]
))

tf.summary.scalar("loss", loss)

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

summary = tf.summary.merge_all()

with tf.Session() as sess:

    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)
    global_step = 0

    sess.run(tf.global_variables_initializer())

    for i in range(50):
        loss_val, _ = sess.run([loss, train], feed_dict={X: X_data, y: y_data})
        print(i, loss_val)

        s, _ = sess.run([summary, train], feed_dict={X: X_data, y: y_data})
        writer.add_summary(s, global_step=global_step)
        global_step += 1
