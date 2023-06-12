import tensorflow as tf
# Define Dense object which is reusable
my_dense = tf.layers.Dense(1, name="optional_name")

# Define some inputs
x1 = tf.constant([[1,1, 1], [2, 2, 2]], dtype=tf.float32)
x2 = tf.constant([[2, 2, 2], [1, 1, 1]], dtype=tf.float32)

# Use the Dense layer
y1 = my_dense(x1)
y2 = my_dense(x2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y1 = sess.run(y1)
    y2 = sess.run(y2)
    print(y1)
    print(y2)