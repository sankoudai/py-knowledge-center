import logging
import logging.handlers

__author__ = 'quiet road'
import unittest


class LoggerTest(unittest.TestCase):
    def setUp(self):
        #设置滚动日志handler
        handler = logging.handlers.TimedRotatingFileHandler('logging_test.log', when='midnight')
        handler.suffix = ".%Y-%m-%d"

        #日志记录格式
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)

        #获取logger
        logger = logging.getLogger('logging_test')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self.logger = logger

    def log_test(self):
        self.logger.debug("Hello, logging")
        self.logger.debug("Hello, logging %s", 'hoho..')

