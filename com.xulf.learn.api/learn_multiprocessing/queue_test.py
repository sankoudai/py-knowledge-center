from queue import Full, Empty
from time import time

__author__ = 'admin'
import multiprocessing as mp
import unittest


class QueueTest(unittest.TestCase):
    """
    Queue are thread and process safe
    """

    def setUp(self):
        self.q = mp.Queue(1)

    def put_test(self):
        self.q.put(1)

        try:
            self.q.put(3, block=False)
        except Full as f:
            print('full at %s' % (time()))

        try:
            self.q.put(3, block=True, timeout=10)
        except Full as f:
            print('full at %s' % (time()))

        self.q.empty()

    def get_test(self):
        self.q.put(2)

        i = self.q.get()
        print("i=%s" % (i))

        try:
            i = self.q.get(block=False)
        except Empty:
            print("empty at %s!" % time())

        try:
            i = self.q.get(block=True, timeout=3)
        except Empty:
            print("empty at %s!" % time())
