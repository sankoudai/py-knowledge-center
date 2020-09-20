__author__ = 'quiet road'
__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class SetTest(unittest.TestCase):
    def setUp(self):
        self.int_set = {1, 2}  # Note: {}-->dict
        self.mixed_set = set(['1', 1, 1, 2])
        self.empty_set = set()

    def test_type(self):
        print(isinstance(self.int_set, set))

    def test_value(self):
        printVar(self.int_set)
        printVar(self.mixed_set)
        printVar(self.empty_set)
    
    def test_len(self):
        print(len(self.mixed_set))
        
    def test_add(self):
        self.int_set.add(3)
        printVar(self.int_set)

        self.int_set.update({4, 5})
        self.int_set.update({6,7}, {8,9})
        printVar(self.int_set)

        self.int_set.update([10, 11])
        printVar(self.int_set)

    def test_remove(self):
        # remove: remove nonexist element results in exception
        self.int_set.remove(1)
        printVar(self.int_set)

        # discard: discart nonexist element is no-op
        self.int_set.discard(1)
        printVar(self.int_set)

        # clear: empty the set
        self.int_set.clear()
        printVar(self.int_set)

        # pop: pop noexist element results in exception

    def test_set_operation(self):
        local_set = {2, 3, 4, 5}
        # union
        a_set = local_set.union(self.int_set)
        printVar(a_set)
        # intersection
        a_set = local_set.intersection(self.int_set)
        printVar(a_set)
        # symmetric difference
        a_set = local_set.symmetric_difference(self.int_set)
        printVar(a_set)
        # difference
        a_set = local_set.difference(self.int_set)
        printVar(a_set)
    
    def test_relation(self):
        # in
        print(1 in self.int_set)

        local_set = {1, 2, 3}
        # issubset
        print(self.int_set.issubset(local_set))
        # issuperset
        print(self.int_set.issuperset(local_set))
        # isdisjoint
        print(self.int_set.isdisjoint(local_set))

    def test_bool_context(self):
        # empty set evaluates to false
        if not set():
            print("empty set evaluates to False")
        if {1}:
            print("nonempty set evaluates to True")

if __name__ == '__main___' :
    unittest.main()
