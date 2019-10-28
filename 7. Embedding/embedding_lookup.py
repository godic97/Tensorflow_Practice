import tensorflow as tf
import numpy as np

vacab_size = 100
embedding_size = 25

inputs = tf.placeholder(tf.int32, shape=[None])

embedding = tf.Variable(tf.random_normal([vacab_size, embedding_size]),dtype=tf.float32)

embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_data = np.array([7, 13, 4,12, 56])
print("Embeeding 전 인풋 데이터: ")
print(sess.run(tf.one_hot(input_data, vacab_size)))
print(tf.one_hot(input_data, vacab_size).shape)

print("Embedding result: ")
print(sess.run([embedded_inputs], feed_dict={inputs : input_data}))
print(sess.run([embedded_inputs], feed_dict={inputs: input_data})[0].shape)