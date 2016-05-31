__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class StringTest(unittest.TestCase):
    """ string can be defined with either single or double quotes
    """

    def setUp(self):
        self.str = "The sky is cloudy"

    def type_test(self):
        print(isinstance(self.str, str))

    def len_test(self):
        printVar(len(self.str))

    def split_test(self):
        words = self.str.split(" ")
        printVar(words)

    def slice_test(self):
        """ start included, end excluded
        """
        seq = '0123456789'
        print(seq[0:3])

    def append_test(self):
        seq = '012' + '3'
        print(seq)

        seq += str(4)
        print(seq)

    def count_test(self):
        seq = 'abcab'
        print(seq.count('ab'))

    def index_test(self):
        """ first occurrence of string
        """
        rul = '0123456'
        seq = '0123433'

        print(seq.index('3'))
        print(seq.rindex('3'))

    def format_placeholder_test(self):
        kb = "KB"
        mb = "MB"
        tempt_str = "1000{0} = 1{1}".format(kb, mb)
        printVar(tempt_str)
        tempt_str = "1000{} = 1{}".format(kb, mb)  # equivalent to previous expression
        printVar(tempt_str)
        tempt_str = '{1}: {1}=1024{0}'.format(kb, mb)
        printVar(tempt_str)

    def format_list_item_test(self):
        li = ['KB', 'MB']
        tempt_str = "1024{0[0]} = 1{0[1]}".format(li)
        printVar(tempt_str)

    def format_dict_item_test(self):
        """Lookup by __getitem method """
        dic = {"name": "Li Lei", "age": 19}
        tempt_str = '{0[name]} is the child'.format(dic)
        printVar(tempt_str)

    def format_object_attribute_test(self):
        """Lookup throught getattr method:  appended attributes can't be resolved"""
        obj = 3 - 4j
        obj.size = 5
        tempt_str = '{0} has imaginary part :{0.imag}'.format(obj)
        print(tempt_str)

    def format_named_argument_test(self):
        tempt_str = '{name} is {age} years old'.format(name='jim', age=16)
        print(tempt_str)

    def format_left_align_test(self):
        tempt_str = "name:{:<15} age:{:<4}"
        print(tempt_str.format('jim', 15))
        print(tempt_str.format('jimmy', 15))

    def format_float_precision_test(self):
        fl = 1.123889
        print("float number: unformated={0}, formated={0:.3f}".format(fl))


if __name__ == '__main___':
    unittest.main()
