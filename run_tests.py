from os import listdir
from os.path import isfile, join
from mutate import run_mutation
import subprocess
import re
import file_copy

# Dictionary of dictionaries with test suite name as the key.
# Each dictionary will store the data for each test suite.
record = {}
# The number of mutations to perform
NUMBER_MUTATIONS = 10
print("Number of mutations to perform: " + str(NUMBER_MUTATIONS))

# TODO: Create the test suites

# Path to where the tests are kept
test_path = "tests/"
# Names of all files in test_path that start with 'test'
test_list = [t for t in listdir(test_path) if (isfile(join(test_path, t)) and t.startswith('test', 0, 4))]
# Number of test files
test_number = len(test_list)

# Source code to test
# MUST BE SET FOR EACH PROGRAM
source_folder = "test_source"
source_path = source_folder + "/"

# takes a string (output from running coverage.py) and returns an int
# representing rounded percentage statement coverage
def parse_coverage_results(s):
    # get last percentage in string
    percent_string = (re.findall("[0-9]*%$", s))[0]
    # take the percentage off and convert to int
    return int(percent_string[:-1])

# takes a string (output from running test cases) and returns a tuple (f,p)
# where f is number failed and p is number passed
def parse_test_results(s):
    try:
        failed_string = (re.findall("[0-9]* failed", s))[0]
        failed_no = int((re.findall("[0-9]*", failed_string))[0])
    except:
        failed_no = 0
    try:
        passed_string = (re.findall("[0-9]* passed", s))[0]
        passed_no = int((re.findall("[0-9]*", passed_string))[0])
    except:
        passed_no = 0
    return (failed_no, passed_no)

# returns the number of test cases in a file
# assumes that each test case starts with "def test_"
def count_test_cases(fname):
    file = open(fname)
    program = file.read()
    tests = program.split("def test_")
    # -1 because the first token item in the splitted list isn't a test
    return len(tests) - 1


# This is the main loop. On the first round, it gets the coverage results for
# each test suite and stores them. It also saves the source code to a different
# location before it is mutated.
c = 0
while c <= NUMBER_MUTATIONS:
    # Run coverage and get benchmark test results on initial source code
    if c < 1:
        print("FIRST ROUND: Collect coverage information")
        # Copy the source code over to a temporary folder
        file_copy.copy_to_temp(source_path)
        for suite_fname in test_list:
            suite_file = test_path + suite_fname
            print("Suite file path: " + suite_file)
            # If the first round, calculate the coverage for the test suite
            # Run coverage on the initial un-mutated files
            print("Running coverage on: " + suite_file)
            try:
                # Run the coverage process on the test suite
                # subprocess.check_output(["coverage","run","--source",source_folder,"-m","py.test", suite_file])
                subprocess.check_output(["coverage","run","--source",source_folder,"-m","py.test", suite_file])
            except subprocess.CalledProcessError as e:
                # Failure encountered
                # print(e.output.decode("utf-8"))
                pass
            # Get the output of the coverage process
            coverage_output = subprocess.check_output(["coverage","report"])
            print("Coverage report")
            print(coverage_output)
            # Parse the coverage results
            coverage = parse_coverage_results(coverage_output.decode("utf-8"))
            print("Coverage: " + str(coverage))
            # Get the number of tests
            num_tests = count_test_cases(suite_file)
            print("Number of test cases in suite: " + str(num_tests))
            
            print("Running tests on: " + suite_file)
            # Run the test cases on initial source code to get benchmark results
            try:
                test_result = subprocess.check_output(["py.test",suite_fname],cwd='tests')
            except subprocess.CalledProcessError as e:
                test_result = e.output.decode("utf-8")
            
            print("Test results:")
            print(test_result)
            
            # Bench results
            bench_results = parse_test_results(test_result)
            bench_failed = bench_results[0]
            bench_passed = bench_results[1]
            print("Benchmark failed tests: " + str(bench_failed))
            print("Benchmark passed tests: " + str(bench_passed))
            
            # Dictionary to store results for each test suite
            result = {}
            # Add results
            result['coverage'] = coverage
            result['num_tests'] = num_tests
            result['bench_failed'] = bench_failed
            result['bench_passed'] = bench_passed
            result['num_muts_caught'] = 0
            result['num_muts'] = 0
            

            # Record result
            record[suite_fname] = result

    # Do mutation and collect test results
    else:
        print("----------Mutation round: " + str(c) + " -----------------")
        # This attempts to mutated the source code
        mutation_performed = run_mutation(source_folder)
        print("Mutation performed: " + str(mutation_performed))
        # Run each test suite we have created
        for suite_fname in test_list:
            # Increment the count of mutations
            if mutation_performed:
                record[suite_fname]['num_muts'] = record[suite_fname]['num_muts'] + 1
            # Run tests using subprocess
            print("Running tests on: " + suite_file)
            try:
                test_result = subprocess.check_output(["py.test",suite_fname],cwd='tests')
            except subprocess.CalledProcessError as e:
                test_result = e.output.decode("utf-8")
            
            print("Test results:")
            print(test_result)
            
            mutated_results = parse_test_results(test_result)
            print("Parsed results on mutated code: " + str(mutated_results))
            mutated_failed = mutated_results[0]
            
            if (mutated_failed > record[suite_fname]['bench_failed']):
                # Mutant caught
                record[suite_fname]['num_muts_caught'] = record[suite_fname]['num_muts_caught'] + 1
                print("Mutant caught! Mutant count: " + str(record[suite_fname]['num_muts_caught']))

    c += 1
    # Copy source code back
    file_copy.restore_from_temp(source_path)

print(record)
