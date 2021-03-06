import numpy as np
import pandas as pd
import tensorflow as tf

n_observations = int(10000)

feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b'a', b'b', b'c', b'd', b'e'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

features_dataset = tf.data.Dataset.from_tensor_slices(
    (feature0, feature1, feature2, feature3)
)
features_dataset

for f0, f1, f2, f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


serialized_example = serialize_example(False, 4, b'c', 0.1234)
serialized_example

example_returned = tf.train.Example.FromString(serialized_example)
example_returned
type(example_returned)


    


