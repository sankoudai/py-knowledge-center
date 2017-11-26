import unittest
import re

class RegexTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def search_test(self):
        res = re.search("(.*)a(.*)", 'sheash')
        print(res.group(0))
        print(res.group(1))
        print(res.group(2))