__author__ = 'quiet road'
import unittest
import time


class DecoratorTest(unittest.TestCase):

    def setUp(self):
        def costly_no_param_func():
            for i in range(3):
                time.sleep(1)

        def costly_fixed_param_func(arg1, arg2):
            print("costly_fixed_param_func: arg1={}, arg2={}".format(arg1, arg2))
            for i in range(3):
                time.sleep(1)

        def no_param_func_decorator(func):
            def wrapper():
                start = time.clock()
                func()
                end = time.clock()
                print("Time cost of {}: {:.2f} seconds (wall-time)".format(func.__name__, end - start))

            return wrapper

        def fixed_param_func_decorator(func):
            def wrapper(arg1, arg2):
                start = time.clock()
                func(arg1, arg2)
                end = time.clock()
                print("Time cost of {}: {:.2f} seconds (wall-time)".format(func.__name__, end - start))

            return wrapper

        def general_param_func_decorator(func):
            def wrapper(*tuple_param, **dict_param):
                start = time.clock()
                func(*tuple_param, **dict_param)
                end = time.clock()
                print("Time cost of {}: {:.2f} seconds (wall-time)".format(func.__name__, end - start))

            return wrapper

        def with_param_decorator(param):
            def __decorator(func):
                def wrapper(*tuple_param, **dict_param):
                    start = time.clock()
                    func(*tuple_param, **dict_param)
                    end = time.clock()
                    print(
                        "Time cost of {}: {:.2f} seconds (wall-time), env={}".format(func.__name__, end - start, param))

                return wrapper

            return __decorator

        self.costly_no_param_func = costly_no_param_func
        self.costly_fixed_param_func = costly_fixed_param_func
        self.no_param_decorator = no_param_func_decorator
        self.fixed_param_func_decorator = fixed_param_func_decorator
        self.general_param_func_decorator = general_param_func_decorator
        self.with_param_decorator = with_param_decorator

    def use_decorator_test(self):
        no_param_func = self.no_param_decorator(self.costly_no_param_func)
        no_param_func()

        fixed_param_func = self.fixed_param_func_decorator(self.costly_fixed_param_func)
        fixed_param_func(1, 2)

        print()
        func = self.general_param_func_decorator(self.costly_no_param_func)
        func()
        func = self.general_param_func_decorator(self.costly_fixed_param_func)
        func(3, 4)

        print()
        func = self.with_param_decorator("test")(self.costly_no_param_func)
        func()
    
    def decorator_grammar_sugar_test(self):
        """Equivalent to local_func = self.no_param_decorator(local_func)"""
        @self.no_param_decorator
        def local_func():
            for i in range(3):
                time.sleep(1)

        local_func()

if __name__ == '__main__':
    unittest.main()
