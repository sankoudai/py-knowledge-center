import unittest
import json


class JsonTest(unittest.TestCase):
    def setUp(self):
        self.json_string = '{"first_name": "Guido", "last_name":"Rossum"}'

    def test_load(self):
        parsed_json = json.loads(self.json_string)
        print(parsed_json)

    def test_attr(self):
        parsed_json = json.loads(self.json_string)
        print(parsed_json['first_name'])

    def test_iter(self):
        parsed_json = json.loads(self.json_string)
        for attr in parsed_json:
            print(attr, parsed_json[attr])

if __name__ == '__main__':
    unittest.main()
