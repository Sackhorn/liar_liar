import os
import unittest
# from teamcity import is_running_under_teamcity
# from teamcity.unittestpy import TeamcityTestRunner
if __name__ == "__main__":
    loader = unittest.TestLoader()
    path = os.path.dirname(__file__)
    tests = loader.discover(path)
    # if is_running_under_teamcity():
    #     testRunner = TeamcityTestRunner()
    # else:
    testRunner = unittest.runner.TextTestRunner()
    testRunner.run(tests)