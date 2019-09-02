import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def build_CNN_classifier(x):
    #MNIST 데이터를 3차원으로 reshape함. 데이터가 grayscale(회색)이기때문에 3번째 차원인 컬러채널의 값은 1
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 1번째 컨볼루션층
    # 5x5 커널 사이즈의 필터 32개
    # 28x28x1 -> 28x28x32
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

    # 1번째 풀링층
    # Max Pooling, 이미지의 크기를 1/2로 downsaple
    # 28x28x32 -> 14x14x32
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2번째 컨볼루션 층
    # 5x5 필터 64개
    # 14x14x32 -> 14x14x64
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

    # 2번째 풀링층
    # 7x7x64
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Fully Connected Layer
    # 7x7인 64개의 Activation map을 1024개의 특징들로 변환
    # 7x7x64(3136) -> 1024
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 출력층
    # 1024개의 특징들을 10개의 클래스로 변환
    # 1024 -> 10
    W_output = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=5e-2))
    b_output = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1, W_output) + b_output
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

y_pred, logits = build_CNN_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
            print("반복: %d, 트레이닝 데이터 정확도: %f" %(i, train_accuracy))
        sess.run([train_step], feed_dict={x : batch[0], y: batch[1]})

    print("테스트 데이터 정확도 : %f" % accuracy.eval(feed_dict={x: mnist.test.images, y:mnist.test.labels}))