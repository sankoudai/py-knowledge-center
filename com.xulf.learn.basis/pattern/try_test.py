__author__ = 'quiet road'
import unittest
from printutil.printutil import printVar

class TryTest(unittest.TestCase):
    def setUp(self):
        class UserException(Exception):
            def __init__(self):
                self.name = "user exception"

            def __str__(self):
                return self.name
        self.user_exception = UserException();
    
    def try_except_test(self):
        ''' uncatched exceptions will propagate.
            unhandled exceptions will terminate program.
        '''
        try:
            print(a)
        except NameError as e:
            print(e)
        else:
            print("fine...")

        print("If exception is catched, program will continue")

    def try_except_finally_test(self):
        '''The finally block will be executed, whether exception is raised or not
            Even there is exit() in try and except block
        '''
        try:
            print(a)
        except NameError as e:
            print(e)
            exit(1)
        finally:
            print('End of try_finally')

        print("You should not be here..")

    def raise_test(self):
        raise NameError("test name error")
        print("You should not be here..")

    def user_defined_exception_test(self):
        raise self.user_exception

if __name__ == '__main__':
    unittest.main()