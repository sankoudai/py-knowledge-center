import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import pandas as pd

import unittest

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


class TestMixedType(unittest.TestCase):
    def setUp(self) -> None:
        train_df = pd.read_csv('data/titanic/train.csv', sep=',')

        self.features = train_df.copy()
        self.labels = self.features.pop('survived')

    def test_mix_preprocess(self):
        # functional api
        # inputs
        inputs = {}
        for name, feat in self.features.items():
            dtype = tf.string if feat.dtype == object else tf.float32
            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        numeric_inputs = {name:input for name, input in inputs.items()
                            if input.dtype == tf.float32}

        string_inputs = {name:input for name, input in inputs.items()
                         if input.dtype == tf.string}

        # 分别处理
        x = layers.Concatenate()(list(numeric_inputs.values()))
        normalize_layer = preprocessing.Normalization()
        normalize_layer.adapt(np.array(self.features[numeric_inputs.keys()]))
        preprocessed_numberic_inputs = normalize_layer(x)

        preprocessed_string_inputs = []
        for name, input in string_inputs.items():
            lookup_layer = preprocessing.StringLookup(vocabulary=np.unique(self.features[name]))
            onehot_encoder_layer = preprocessing.CategoryEncoding(max_tokens=lookup_layer.vocab_size())

            x = lookup_layer(input)
            x = onehot_encoder_layer(x)
            preprocessed_string_inputs.append(x)

        # 合并
        preprocessed_inputs = [preprocessed_numberic_inputs] + preprocessed_string_inputs
        preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
        preprocessor = tf.keras.Model(inputs=inputs, outputs=preprocessed_inputs_cat)

        # 测试使用
        feature_dict = {name:np.array(value) for name, value in self.features.items()}
        test_case = {name:values[:1] for name, values in feature_dict.items()}
        print(test_case)
        print(preprocessor(test_case))

        def titanic_model(preprocessing_head, inputs):
            body = tf.keras.Sequential([
                layers.Dense(64),
                layers.Dense(1)
            ])

            preprocessed_inputs = preprocessing_head(inputs)
            result = body(preprocessed_inputs)
            model = tf.keras.Model(inputs, result)

            model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=tf.optimizers.Adam())
            return model

        titanic_model = titanic_model(preprocessor, inputs)
        titanic_model.fit(x=feature_dict, y=self.labels, epochs=10)