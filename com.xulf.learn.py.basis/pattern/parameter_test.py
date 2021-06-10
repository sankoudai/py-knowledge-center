__author__ = 'quiet road'
import unittest


class ParameterTest(unittest.TestCase):
    def setUp(self):
        def positional_param_f(arg1, arg2):
            """You must provide all positional arguments, no more no less"""
            print("positional params : arg1={}, arg2={}".format(arg1, arg2))

        def keyword_param_f(name="name", gender=None):
            """Keyword parameter is defined by giving it  a default value.
                Python interpreter won't care order of keyword parameters.
                You can omit some parameter ( Then default value will be used).
            """
            print("keyword params: name={}, gender={}".format(name, gender))

        def mixed_param_f(arg1, arg2, name="name", gender=None):
            """When one defines mixed positional-keyword-parm function,
                positional params must come first and complete, then keyword params.
                When one calls positional-keyword-parm function,
                positional params must come first and complete, then keyword parameters.
            """
            show_string = "mixed positional-keyword params: arg1={}, arg2={}, name={}, gender={}"
            print(show_string.format(arg1, arg2, name, gender))

        def collect_positional_param_f(*tuple_param):
            """Python interpreter will pack positional parameters to a tuple(tuple_param).
                You can pass any number of positional paremeters.
            """
            print("collect positional param: tuple_param={}".format(tuple_param))

        def collect_keyword_param_f(**dict_param):
            """python interpreter will pack named parameters to a dict (dict_parm)"""
            print("collect keyword param: dict_param={}".format(dict_param))

        def collect_mixed_param_f(*tuple_param, **dict_parm):
            print("Collect mixed param: tuple_param={}, dict_param={}".format(tuple_param, dict_parm))

        self.positional_param_f = positional_param_f
        self.keyword_param_f = keyword_param_f
        self.mixed_param_f = mixed_param_f
        self.collect_positional_param_f = collect_positional_param_f
        self.collect_keyword_param_f = collect_keyword_param_f
        self.collect_mixed_param_f = collect_mixed_param_f

    def test_positional_param(self):
        self.positional_param_f(1, 2)

        try:
            self.positional_param_f(1, 2, 3)
        except TypeError:
            print("Call with To many paramters!")

        try:
            self.positional_param_f(1)
        except TypeError:
            print("Call with less parameters!")

    def test_keyword_param(self):
        # order does not matter
        self.keyword_param_f(name="jim", gender="male")
        self.keyword_param_f(gender="male", name="jim")

        # you can omit parameters
        self.keyword_param_f(name="jim")
        self.keyword_param_f()

    def test_mixed_param(self):
        self.mixed_param_f(1, 2, name="jim", gender="male")

        # keyword params must follow positional params: following line won't run
        # self.mixed_param_f(1, name="jim", 2, gender="male")

    def test_collect_positional_param(self):
        self.collect_positional_param_f()
        self.collect_positional_param_f(1)
        self.collect_positional_param_f(1, "one")

    def test_collect_keyword_param(self):
        self.collect_keyword_param_f()
        self.collect_keyword_param_f(name="jim", age=17)

    def test_collect_mixed_param_f(self):
        self.collect_mixed_param_f()
        self.collect_mixed_param_f(1, 2)
        self.collect_mixed_param_f(name="jim", gender="male")
        self.collect_mixed_param_f(1, 2, name="jim")

    def test_unpack_sequence_as_positional_param(self):
        local_seq = (1, 2)
        self.positional_param_f(*local_seq)

        local_seq = [1, 2]
        self.positional_param_f(*local_seq)
    
    def test_unpack_dict_as_keyword_param(self):
        local_dict = {"name": "jim", "gender": "male"}
        self.keyword_param_f(**local_dict)
