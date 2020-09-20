__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class DictTest(unittest.TestCase):
    def setUp(self):
        self.str_key_dict = {"name":"jim", "age":19}
        self.int_key_dict = {1:"jim", 2:"lilei"}
        self.list_val_dict = {1000:['KB', 'MB'], 1024:['KiB', 'MiB']}
        self.empty_dict = {}

    def test_type(self):
        print(isinstance(self.str_key_dict, dict))
        print(isinstance(self.empty_dict, dict))

    def test_value(self):
        printVar(self.str_key_dict)
        printVar(self.int_key_dict)
        printVar(self.list_val_dict)
        printVar(self.empty_dict)

    def test_access(self):
        printVar(self.str_key_dict['name'])

        # nonexist key results in exception
        printVar(self.str_key_dict['gender'])

    def test_modify(self):
        self.str_key_dict['name'] = 'jimmy'
        printVar(self.str_key_dict)

    def test_bool_context(self):
        # empty dict evaluates to false
        if not {}:
            print('Empty dict evaluates to False')
        if {1:'jim'}:
            print('Nonempty dict evaluates to True')

if __name__ == '__main___' :
    unittest.main()
