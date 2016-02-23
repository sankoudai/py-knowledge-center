__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.string_sequence = 'abcd'
        self.list_sequence = [1, 2, 3]
        self.tuple_sequence = (4, 6, 5)
        self.range = range(0, 10, 2)

    def while_test(self):
        name = ''
        while not name:
            name = 'jim' #input("enter your name:")
        print("Name:" + name)

    def loop_sequence_test(self):
        print("-------string sequence-------------")
        for ch in self.string_sequence:
            printVar(ch)

        print("-------list sequence-------------")
        for item in self.list_sequence:
            printVar(item)

        print("-------tuple sequence-------------")
        for item in self.tuple_sequence:
            printVar(item)

        print("-------range-------------")
        for item in self.range:
            printVar(item)

    def loop_dictionary_test(self):
        d = {'one':1, "two":2, "three":3}
        for key in d:
            print("{}:{}".format(key, d[key]))

        print("-------------------------")
        for key, val in d.items():
            print("{}:{}".format(key, val))

    def util_zip_test(self):
        key_list = ["one", "two", "three", "four"]
        val_list = [1, 2, 3]
        for key, val in zip(key_list, val_list):
            print("{}:{}".format(key, val))
    
    def util_enumerate_test(self):
        vals = ["one", "two", "three", "four"]
        for i, val in enumerate(vals):
            print("{}:{}".format(i, val))

    def util_sorted_test(self):
        '''sorted(seq) returns sorted version as a list'''
        local_string = 'bac'
        printVar(sorted(local_string)) # list
        local_tuple = (3, 5, 4)
        for val in (sorted(local_tuple)):
            printVar(val)

    def util_reversed_test(self):
        '''reversed(seq):  returns reversed object'''
        local_string = 'abc'
        printVar(reversed(local_string)) # reversed object
        for ch in reversed(local_string):
            printVar(ch)

    def break_test(self):
        for i in range(10):
            print(i)
            if i == 3:break

    def continue_test(self):
        for i in range(5):
            if i % 3 == 0: continue
            print(i)

if __name__ == '__main__':
    unittest.main()