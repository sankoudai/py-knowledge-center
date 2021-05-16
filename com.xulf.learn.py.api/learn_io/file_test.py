__author__ = 'quiet road'
import unittest
from printutil.printutil import printVar
import _io


class FileTest(unittest.TestCase):
    """file object
        An object exposing file-orienteed API. Depending on how it's created,
        a file object can mediate access to disk file, std IO, sockets, pipes, etc.

        Three category: raw binary file, buffered binary file and text file.
        Also called: file-like object, streams
    """

    def setUp(self):
        self.text_file_read = open("data/file_test.txt", encoding="utf-8")
        self.text_file_trunc = open("data/file_trunc.txt", encoding="utf-8", mode='w')
        self.text_file_append = open("data/file_append.txt", encoding="utf-8", mode='a')

        self.bin_file_read = open("data/pic.bin", mode="rb")
        self.bin_file_write = open("data/file_test_write.bin", mode="wb")

    def test_type(self):
        assert isinstance(self.text_file_read, _io.TextIOWrapper)

    def test_attribute(self):
        print("file attribute: name={}".format(self.text_file_read.name))
        print("file attribute: encoding={}".format(self.text_file_read.encoding))

    def test_text_file_read(self):
        # read by chars
        print('------by char------------')
        self.text_file_read.seek(0)
        chs = self.text_file_read.read(4)
        printVar(chs)

        # read a line (from current cursor)
        print('--------by line----------')
        self.text_file_read.seek(0)
        line = self.text_file_read.readline()  # include line ending
        printVar(line)

        # read in whole file as string
        print('-------by file-----------')
        self.text_file_read.seek(0)
        content = self.text_file_read.read()
        printVar(content)

        # read in whole file as list of lines
        print('-------by file-----------')
        self.text_file_read.seek(0)
        lines = self.text_file_read.readlines()
        printVar(lines)

        # read beyond end of file
        print('-------read EOF-----------')
        self.text_file_read.read()
        ch = self.text_file_read.read(1)
        assert ch == ''

        # close
        self.text_file_read.close()

    def test_text_file_write(self):
        # truncate: create if file does not exist
        self.text_file_trunc.write(" later")

        # append: create if file does not exist
        self.text_file_append.write(" later")

        # by character:
        self.text_file_append.write(" character")

        # by line
        lines = [' hello\n', ' jimmy\n']
        self.text_file_append.writelines(lines)

    def test_bin_file_read(self):
        # byte
        byte = self.bin_file_read.read(1)
        printVar(byte)

        # bytes
        self.bin_file_read.seek(0)
        file_bytes = self.bin_file_read.read()
        printVar(file_bytes)

        self.bin_file_read.close()

    def test_bin_file_write(self):
        self.bin_file_write.write(b"abc")
        self.bin_file_write.write(" abc".encode(encoding="utf-8"))
        self.bin_file_write.close()

    def test_cursor_position(self):
        # tell
        pos = self.text_file_read.tell()
        print("current cursor position: {}th byte".format(pos))
        self.text_file_read.read(10)
        pos = self.text_file_read.tell()
        print("current cursor position: {}th byte".format(pos))

        # seek
        self.text_file_read.seek(0)
        pos = self.text_file_read.tell()
        print("current cursor position: {}th byte".format(pos))

    def test_auto_close(self):
        with open("data/file_test.txt", encoding="utf-8") as f:
            for line in f:
                print(line)

if __name__ == '__main__':
    unittest.main()
