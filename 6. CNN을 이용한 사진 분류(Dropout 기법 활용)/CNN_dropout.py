import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets.cifar10 import load_data

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_CNN_classifier(x):
    x_image = x

    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2)) # 정규 분포에서 난수 생성, shape-> [가로, 세로, 입력채널, 출력채널], stddev->표준편차
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1) # parameter->(input,filter, strides, padding) strides 뛰어넘는거, 1->배치, 2-> 가로, 3->세로, 4->채널, SAME -> input의 모양 그대로 출력(zero padding)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME") #ksize -> 필터 사이즈(3 by 3)

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)

    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding="SAME") + b_conv4)

    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding="SAME") + b_conv5)

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

    h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128]) # 1 by 8 * 8 * 384 행렬, -1이 있으면 모양 알아서 맞춰줌
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

x = tf.placeholder(tf.float32, shape=[None, 32 , 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

y_pred, logits = build_CNN_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

            print("반복: %d, 트레이닝 데이터 정확도: %f 손실함수: %f" % (i, train_accuracy, loss_print))

        sess.run(train_step,feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
        test_accuracy = test_accuracy / 10
        print("테스트 데이터 정확도: %f" % test_accuracy)