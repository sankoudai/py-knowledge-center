import unittest
import argparse

class ArgparseTest(unittest.TestCase):
    def test_add_arguement(self):
        parser = argparse.ArgumentParser(description='test')

        # -自动转化成下划线
        parser.add_argument('--emb-path', type=str, default='/path')
        args = parser.parse_args()
        assert args.emb_path == '/path'
