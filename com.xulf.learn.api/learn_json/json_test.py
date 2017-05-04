import unittest
import json


class JsonTest(unittest.TestCase):
    def setUp(self):
        self.json_string = '{"first_name": "Guido", "last_name":"Rossum"}'

    def load_test(self):
        parsed_json = json.loads(self.json_string)
        print(parsed_json)

    def attr_test(self):
        parsed_json = json.loads(self.json_string)
        print(parsed_json['first_name'])

    def iter_test(self):
        parsed_json = json.loads(self.json_string)
        for attr in parsed_json:
            print(attr, parsed_json[attr])

if __name__ == '__main__':
    unittest.main()
