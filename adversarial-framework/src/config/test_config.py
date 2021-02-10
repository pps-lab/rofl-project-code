import unittest

from config import load_config


class ConfigTest(unittest.TestCase):

    def test_load_config(self):
        load_config("example_config.yaml")


if __name__ == '__main__':
    unittest.main()
