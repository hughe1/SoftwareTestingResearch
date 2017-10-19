from os import listdir
from os.path import isfile, join
from mutate import run_mutation
import subprocess
import pytest

runs = {}
muts = {}

# import create_suite

# Create test suites using test creation script
# TODO: Finish off create_suite()
# create_suite()

# Get test suites 
test_path = "tests/"
test_list = [t for t in listdir(test_path) if isfile(join(test_path, t))]
test_number = len(test_list)

# Run each test suite we have created
# PROBLEM: Unit tests are running over codebase not just the mutated files
for suite in test_list:
    i += 1
    mutations = run_mutation
    test_name = "test" + str(i)
    # Run the unittests 
    # TODO: Only run them over the created mutation files, not the whole
    # codebase
    test = pytest.main(['tests/' + suite,'--result-log=logs/' + suite + ".txt"])
    print("Test results: " + str(test))
    log_file = open("logs/" + suite + ".txt")
    log_lines = log_file.readlines()
    errors = 0
    for line in lines:
        if "AssertionError" in lines:
            errors += 1
    # Store mutations and errors for each test suite in dictionary
    runs[suite] = [mutations, errors]
    return errors