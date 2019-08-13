import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
num_epochs = 30
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

# 입력값, 출력값 설정
x = tf.compat.v1.placeholder(tf.float32, shape=[None, input_size])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, output_size])

# MLP 정의 1개의 입력층, 2개의 은닉층, 1개의 출력층
# 활성함수: ReLU
def build_ANN(x):
    W1 = tf.Variable(tf.random.normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random.normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random.normal(shape=[hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)

    W_output = tf.Variable(tf.random.normal(shape=[hidden2_size, output_size]))
    b_output = tf.Variable(tf.random.normal(shape=[output_size]))
    logits = tf.matmul(H2_output, W_output) + b_output

    return logits

predicted_value = build_ANN(x)

# 손실 함수 정의(소프트맥스 & 크로스 엔트로피)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_value, labels=y)) # 손실함수
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # 손실함수 최적화

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        average_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})

            average_loss += current_loss / total_batch

        if epoch % display_step == 0:
            print("반복: %d, 손실함수(Loss): %f" % ((epoch+1), average_loss))

    # 최종 정확도 측정
    correct_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("정확도(Accuracy): %f" % (accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})))
