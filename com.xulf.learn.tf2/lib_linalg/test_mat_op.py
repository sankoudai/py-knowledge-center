import tensorflow as tf
import unittest
from cmp_util import *


class TestMatOp(unittest.TestCase):
    def test_diag(self):
        # 构造指定对角元素的举矩阵: diag
        #   方法签名：tf.linalg.diag(diagnal, name='diag', k=0, num_rows=-1, num_cols=-1, padding_value=0, align='RIGHT_LEFT')
        #       diagnal: tensor
        #       k: (不起作用!!) 偏移量; k=0则是对角， k>0则是上对角矩阵， k<0则是下对角。k不起作用!!!
        diagnal = tf.constant([1, 2, 3])
        m = tf.linalg.diag(diagnal)
        assert_equal([[1, 0, 0],
                      [0, 2, 0],
                      [0, 0, 3]], m)

        diagnal = tf.constant([[1, 2, 3],
                               [4, 5, 6]])
        m = tf.linalg.diag(diagnal)
        assert_equal([
            [[1, 0, 0],
             [0, 2, 0],
             [0, 0, 3]],

            [[4, 0, 0],
             [0, 5, 0],
             [0, 0, 6]]
        ], m)

        # 给定矩阵， 设置其对角元素: set_diag
        #   方法签名: tf.linalg.set_diag(matrix, diagnal, name='set_diag', k=0, align='RIGHT_LEFT')
        #       matrix: tensor of shape [..., M, N]
        #       k: 偏移量，参考diag
        #       diagnal: tensor
        #       返回值：
        m = tf.fill([2, 3, 4], 7)
        diagnal = tf.constant([
            [1, 2, 3],
            [4, 5, 6]
        ])
        transformed_m = tf.linalg.set_diag(m, diagnal, k=0)
        assert_equal([[[1, 7, 7, 7],
                       [7, 2, 7, 7],
                       [7, 7, 3, 7]],
                      [[4, 7, 7, 7],
                       [7, 5, 7, 7],
                       [7, 7, 6, 7]]],
                     transformed_m)

    def test_inv(self):
        # 逆矩阵:
        #   方法签名: tf.linalg.inv(tensor, name=None)
        #       tensor: Shape is [..., M, M], 最后两维需是可逆矩阵
        #   说明：tensor[..., :,:] 不可逆， 结果不可知

        m = tf.constant([[1, 0, 0],
                         [0, 2, 0],
                         [0, 0, 3]], dtype=tf.float32)
        inv_m = tf.linalg.inv(m)
        assert_equal(inv_m, [[1, 0, 0],
                             [0, 0.5, 0],
                             [0, 0., 0.3333334]], tol=1e-6)

    def test_normalize(self):
        # 归一化
        # 方法签名: tf.linalg.normalize(tensor, ord='euclidean', axis=None, name=None)
        #   axis: None => tensor被认为是单一vector
        #         integer => tensor被认为是一批vectors
        #         2-tuple => tensor被认为是一批matrixs
        #   ord:  对vector， 取值范围为euclidean（2-范数）, 正数 (p-范数), np.inf(无穷范数）
        #         对matrix， 取值范围为fro (等价于euclidean距离), 1, 2, np.inf
        #   返回值：normalized_tensor - 规范化的矩阵， 与tensor具有相同的shape
        #          norm - 沿axis的范数，axis维度大小为1， 其余维度与原矩阵相同大小

        tensor = tf.constant([
            [[1, 2, 7],
             [2, 3, 5],
             [3, 4, 3]],

            [[4, 5, 1],
             [5, -4, 1],
             [4, 0, 6]]
        ], dtype=tf.float32)  # （2, 3, 3)
        # vector的1-范数： 绝对值之和
        normalized_m, norms = tf.linalg.normalize(tensor, ord=1, axis=2)
        assert_equal(normalized_m,
                     [[[0.1, 0.2, 0.7],
                       [0.2, 0.3, 0.5],
                       [0.3, 0.4, 0.3]],

                      [[0.4, 0.5, 0.1],
                       [0.5, -0.4, 0.1],
                       [0.4, 0.0, 0.6]]], tol=1e-6)

        # 矩阵的1-范数： 列绝对值和的最大值
        normalized_m, norms = tf.linalg.normalize(tensor, ord=1, axis=[1, 2])
        assert_equal(normalized_m,
                     [[[1 / 15., 2 / 15., 7 / 15.],
                       [2 / 15., 3 / 15., 5 / 15.],
                       [3 / 15., 4 / 15., 3 / 15.]],

                      [[4 / 13., 5 / 13., 1 / 13.],
                       [5 / 13., -4 / 13., 1 / 13.],
                       [4 / 13., 0 / 13., 6 / 13.]]], tol=1e-6)
        assert_equal(norms, [[[15]], [[13]]])

    def test_statis(self):
        # 行列式: det
        #   方法签名: tf.linalg.det(m, name=None)
        #       m:  m[.., M, M]， 最后2维决定矩阵，..指定batch
        #    返回值:  c[..]
        m = tf.constant([[1, 2],
                         [3, 4]], dtype=tf.float32)
        assert_equal(-2, tf.linalg.det(m))
