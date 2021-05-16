import tensorflow as tf

a= tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
with tf.Session() as sess:
    v = sess.run(x)
    print(v)

writer.close()