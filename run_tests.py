from os import listdir
from os.path import isfile, join
from mutate import run_mutation
import subprocess

# List of dictionaries. Each dictionary will store the data for each test suite.
record = []
# The number of mutations to perform
NUMBER_MUTATIONS = 0

# TODO: Create the test suites

# Get test suites 
test_path = "tests/"
test_list = [t for t in listdir(test_path) if isfile(join(test_path, t))]
test_number = len(test_list)

# Source code to test
# MUST BE SET FOR EACH PROGRAM
source_folder = "test_source"

# takes a string (output from running coverage.py) and returns an int
# representing rounded percentage statement coverage
def parse_coverage_results(s):

    # get last percentage in string
    percent_string = (re.findall(“[0-9]*%$“, s))[0]

    # take the percentage off and convert to int
    return int(percent_string[:-1])

# takes a string (output from running test cases) and returns a tuple (f,p)
# where f is number failed and p is number passed
def parse_test_results(s):
    failed_string = (re.findall(“[0-9]* failed”, s))[0]
    passed_string = (re.findall(“[0-9]* passed”, s))[0]
    failed_no = int((re.findall(“[0-9]*“, failed_string))[0])
    passed_no = int((re.findall(“[0-9]*“, passed_string))[0])
    return (failed_no, passed_no)


# This is the main loop. On the first round, it gets the coverage results for
# each test suite and stores them. It also saves the source code to a different
# location before it is mutated.
c = 0
while c <= NUMBER_MUTATIONS:
    
    # Run coverage and get benchmark test results on initial source code
    if c < 1:
        for suite in test_list:    
            # If the first round, calculate the coverage for the test suite
            # TODO: Copy the source code over to a different file
            # Run coverage on the initial un-mutated files
            try:
                subprocess.check_output(["coverage","run","--source",source_folder,"-m","py.test","tests/test_1.py"])
            except subprocess.CalledProcessError as e:
                print e.output
            coverage_output = subprocess.check_output(["coverage","report"])
            # Parse the coverage results 
            coverage = parse_coverage_results(coverage_output)
            # TODO: Store coverage results
    
    # Do mutation and collect test results
    else:
        # This mutates the source code and gets the number of mutations
        num_mutations = run_mutation
        # Run each test suite we have created
        # PROBLEM: Unit tests are running over codebase not just the mutated files
        i = 0
        for suite in test_list:
            # Get the test name
            test_name = "test_" + str(i) + ".py"
            # Run tests using subprocess
            try:
                test_result = subprocess.check_output(["py.test","test_1.py"],cwd='tests')
            except subprocess.CalledProcessError as e:
                print e.output
            i += 1
            
            # TODO: Update records data structure with test results for suite
            # Store mutations and errors for each test suite in dictionary
            # entry["test_name"] = test_name
            # entry["num_tests"] = num_tests
            # entry["num_muts_caught"] = num_muts_caught
            # entry["total_muts"] = total_muts
            # entry["mut_score"] = mut_score

    c += 1
    # TODO: Copy source code back
    