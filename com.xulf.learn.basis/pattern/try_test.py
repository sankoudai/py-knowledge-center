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
        self.user_exception = UserException()
    
    def test_try_except(self):
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

    def test_try_except_finally(self):
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

    def test_raise(self):
        raise NameError("test name error")
        print("You should not be here..")

    def test_user_defined_exception(self):
        raise self.user_exception

if __name__ == '__main__':
    unittest.main()