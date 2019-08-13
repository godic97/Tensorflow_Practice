import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.02
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10
input_size = 784
hidden1_size = 128
hidden2_size  = 64

x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, 10])

def build_autoencoder(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1_output = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2_output = tf.nn.sigmoid(tf.matmul(H1_output, W2) + b2)

    W3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]))
    b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H3_output = tf.nn.sigmoid(tf.matmul(H2_output, W3) + b3)

    W4 = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]))
    b4 = tf.Variable(tf.random_normal(shape=[input_size]))
    reconstructed_x = tf.nn.sigmoid(tf.matmul(H3_output, W4) + b4)

    return reconstructed_x, H2_output

def build_softmax_classifier(x):
    W_softmax = tf.Variable(tf.zeros([hidden2_size, 10]))
    b_softmax = tf.Variable(tf.zeros([10]))
    y_pred = tf.nn.softmax(tf.matmul(x, W_softmax) + b_softmax)

    return y_pred

y_pred , extracted_features = build_autoencoder(x)
y_true = x
y_pred_softmax = build_softmax_classifier(extracted_features)

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

finetuning_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred_softmax), reduction_indices=[1]))
finetuning_train_step = tf.train.GradientDescentOptimizer(0.5).minimize(finetuning_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)

#오토인코더를 활요한 데이터 정제
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_xs})

            if epoch % display_step == 0:
                print("반복: %d, loss: %f" %((epoch+1), current_loss))
    print("오토인코딩 완료")

#오토인코더로 정제한 데이터를 활용한 숫자 분류기
    for epoch in range(training_epochs + 100):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, finetuning_loss_print = sess.run([finetuning_train_step, finetuning_loss], feed_dict={x: batch_xs, y: batch_ys})
            if epoch % display_step == 0:
                print("반복: %d, loss: %f" %((epoch+1), finetuning_loss_print))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred_softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy: %f" %sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))