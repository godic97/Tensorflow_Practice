import tensorflow as tf


W = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))
x = tf.placeholder(tf.float32)
logits = W*x + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.square(logits - y))

#train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

grad_clip = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
train_step = tf.train.GradientDescentOptimizer(0.01).apply_gradients(zip(grads, tvars))

x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_train, y: y_train})

x_test = [3.5, 2.5, 1.5, 7.5]
print(sess.run(logits, feed_dict={x: x_test}))

sess.close()
