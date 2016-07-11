import unittest
from datetime import datetime

from printutil.printutil import printVar


class DatetimeTest(unittest.TestCase):
    def setUp(self):
        self.pattern_pair = []
        self.pattern_pair.append(('2016-05-27', '%Y-%m-%d'))
        self.pattern_pair.append(('25/May/2016:00:00:01 +0800', '%d/%b/%Y:%H:%M:%S %z'))
        self.pattern_pair.append(('25/May/2016:00:00:01', '%d/%b/%Y:%H:%M:%S'))

    def conversion_test(self):
        for date_str, fmt_str in self.pattern_pair:
            dt = datetime.strptime(date_str, fmt_str)

            print('date_str={},  fmt_str={}'.format(date_str, fmt_str))
            printVar(dt)
            print()

    def millis_datetime_conversion_test(self):
        ms = 1467075826837
        dt = datetime.fromtimestamp(ms/1000.0)
        printVar(dt, '%Y-%m-%d %H:%M:%S')


