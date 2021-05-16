__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class BytesTest(unittest.TestCase):
    """bytes is like sequence of immutable int, with range [0, 256)"""
    def setUp(self):
        self.single_quote_bytes = b'allow double quotes""'
        self.double_quote_bytes = b"allow single quotes''"
        self.hexadecimal_bytes = b'\xff'

    def test_type(self):
        assert isinstance(self.single_quote_bytes, bytes)

    def test_value(self):
        printVar(self.single_quote_bytes)
        printVar(self.hexadecimal_bytes)

    def test_int_conversion(self):
        # to int
        i = self.single_quote_bytes[0]
        printVar(i)

        # from int
        b = bytes([15])
        printVar(b)
    
    def test_string_conversion(self):
        string = "012345"
        # string to bytes
        b = bytes(string, encoding="utf-8")
        printVar(b)
        b = string.encode(encoding="utf-8")
        printVar(b)

        # bytes to string
        string = str(b, encoding="utf-8")
        printVar(string)
        string = b.decode(encoding="utf-8")
        printVar(string)


if __name__ == '__main___':
    unittest.main()
