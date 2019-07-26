import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets("/tmp/data", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
logits = tf. matmul(x, w) - b
y_pred = tf.nn.softmax(logits)

# 참값에 해당하는 확률만 남긴 후(onehot인코딩으로 참값에 해당하는 확률만 남고 차원이 하나 줄어듬)
# 그거를 다시 평균냄(log의 진수가 1(100%를 의미)에서 멀어지면 멀어질수록 값이 상승함 따라서 손실함수의 결과값 상승)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = minist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)) #같으면1?

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #나온 정답들의 평균을 구함
print("정확도: %f" % sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels}))

sess.close()