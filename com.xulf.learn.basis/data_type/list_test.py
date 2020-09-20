__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class ListTest(unittest.TestCase):
    def setUp(self):
        self.string_list = ['summer', 'is', 'at', 'its', 'uttermost']
        self.int_list = [2, 3, 7, 12]
        self.mixed_list = ['spring', 1, 'summer', 2, 'autumn', 3, 'winter', 4]

    def test_type(self):
        print(isinstance(self.string_list, list))

    def test_slice(self):
        '''slicing: index forms circle'''
        printVar(self.string_list)
        printVar(self.string_list[0]) # first
        printVar(self.string_list[-1]) # last

        # equivalent
        printVar(self.string_list[0:3])
        printVar(self.string_list[:3])

    def test_slice_relation(self):
        '''Slicing creates a new list'''
        slice_part = self.string_list[0:2]
        print("slice_part before modify -- {}".format(slice_part))
        print("original list after modity -- {}".format(self.string_list))

        slice_part[0] = 'jim'
        print("slice_part after modify -- {}".format(slice_part))
        print("original list after modity -- {}".format(self.string_list))

        print('-----Conclusion: slicing creates a new list-----')

    def test_add_item(self):
        a_list = [1, 2 ,3]

        # + creates a second list
        b_list = a_list + [4]
        print("a_list: {}".format(a_list)) # [1, 2, 3]
        print('b_list: {}'.format(b_list)) # [1, 2, 3, 4]
        print()

        # append,insert, extend modifies list
        a_list.append(4)
        print("a_list: {}".format(a_list)) # [1, 2, 3, 4]

        a_list.insert(0, 0)
        print("a_list: {}".format(a_list)) # [0, 1, 2, 3, 4]

        a_list.extend([5, 6])
        print("a_list: {}".format(a_list)) # [0, 1, 2, 3, 4, 5, 6]

    def test_search_item(self):
        a_list = ['new', 'world', 'new']

        # in
        printVar('new' in a_list)

        # count
        printVar(a_list.count('new'))

        # index: raise exception if nonexist
        printVar(a_list.index('new'))
        a_list.index('a') # raise exception

    def test_remove_item(self):
        a_list = ['new', 'world', 'new', 'life', 'new', 'beginning']

        # del by index
        del a_list[1]
        printVar(a_list)
        # remove by value
        a_list.remove('life')
        printVar(a_list)
        print()

        #pop
        val = a_list.pop()
        print("val:{}, a_list:{}".format(val, a_list))
        val = a_list.pop(1)
        print("val:{}, a_list:{}".format(val, a_list))

    def test_bool_context(self):
        # empty list is false
        if(not []):
          print("Empty list is False in boolean context")

        print("All other list is True in boolean context")

if __name__ == '__main___' :
    unittest.main()
