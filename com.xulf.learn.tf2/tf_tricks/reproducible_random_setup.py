'''
    可复现的keras程序环境
    参考: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
'''
import numpy as np
import tensorflow as tf
import random as python_random
import os

# For hash-based algorithm
os.environ['PYTHONHASHSEED'] = '0'

# 不使用GPU: GPU的并发可能造成部分不确定性
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

