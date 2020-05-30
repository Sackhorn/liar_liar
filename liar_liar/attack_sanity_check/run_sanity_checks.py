import os
import unittest
loader = unittest.TestLoader()
path = os.path.dirname(__file__)
tests = loader.discover(path)
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)