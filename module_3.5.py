import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset1.element_spec

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
dataset2.element_spec

dataset3 = tf.data.Dataset.zip(dataset1(dataset1, dataset2))
dataset3.element_spec

dataset = tf.data.Dataset.range(100)
dataset

dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset