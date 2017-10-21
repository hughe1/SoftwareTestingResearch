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
NUMBER_MUTATIONS = 4
print("Number of mutations to perform: " + str(NUMBER_MUTATIONS))
OUTPUT_FILENAME = "output.csv"

# TODO: Create the test suites. This currently has to be done manually.
# subprocess.check_output(["python","divide_test_suite.py","tests"])

# Path to where the tests are kept. All tests must be placed in this folder
# manually.
# NOTE: Use the same folder name as the repository of the program under test
# uses. Replace the tests that are in there with the create test suites.
test_path = "tests/"

# TODO: Changes to new_tests if divide_test_suite is implemented
# test_path = "new_tests/"

# Names of all files in test_path that start with 'test'
test_list = [t for t in listdir(test_path) if (isfile(join(test_path, t)) and t.startswith('test', 0, 4))]
# Number of test files
test_number = len(test_list)

# Source code to test
# NOTE: MUST BE SET MANUALLY FOR EACH PROGRAM. This is just an example.
source_folder = "httpie"
source_path = source_folder + "/"

# Takes a string (output from running coverage.py) and returns an int
# Representing rounded percentage statement coverage.
def parse_coverage_results(s):
    # get last percentage in string
    percent_string = (re.findall("[0-9]*%$", s))[0]
    # take the percentage off and convert to int
    return int(percent_string[:-1])

# Takes a string (output from running test cases) and returns a triple
def parse_test_results(s):
    try:
        failed_list = (re.findall("[0-9]* failed", s))
        failed_string = failed_list[len(failed_list)-1]
        failed_no = int((re.findall("[0-9]*", failed_string))[0])
    except Exception, e:
        print("EXCEPTION - Parse 'failed':" + str(e))
        failed_no = 0
    try:
        passed_list = (re.findall("[0-9]* passed", s))
        passed_string = passed_list[len(passed_list)-1]
        passed_no = int((re.findall("[0-9]*", passed_string))[0])
    except Exception, e:
        print("EXCEPTION - Parse 'passed':" + str(e))
        passed_no = 0
    try:
        error_string = (re.findall("[0-9]* error in", s))[0]
        error_no = int((re.findall("[0-9]*", error_string))[0])            
    except Exception, e:
        # print("EXCEPTION - Parse 'errors':" + str(e))
        error_no = 0
    return (failed_no, passed_no, error_no)

# Returns the number of test cases in a file.
# Assumes that each test case starts with "def test_".
def count_test_cases(fname):
    file = open(fname)
    program = file.read()
    tests = program.split("def test_")
    # -1 because the first token item in the splitted list isn't a test
    return len(tests) - 1

# Saves output of test results to a csv file
# Takes a dictionary of dictionaries, one representing each test suite and its
# statistics.
# Output csv has a header for each key in each sub-dictionary
def save_output(d):
	#initialise output string
	s = ""
	# print headers
	headers = ["test_suite"] + d[d.keys()[0]].keys()
	for header in headers:
		s = s + header + ","
	# print each row
	for test_suite in d.keys():
		s+='\n'
		# add test_suite name to the row
		s += test_suite + ","
		# add all the header values to the row
		row = d[test_suite]
		for header in headers[1:]:
			s += str(row[header]) + ","
	# save the data to a file
	output_file = open(OUTPUT_FILENAME,'w')
	output_file.write(s)


# This is the main loop. On the first round, it gets the coverage results for
# each test suite and stores them. It also saves a copy of the source code in a
# temporary location. 
# After the first loop, it mutates the source code and collects the results.
# Each iteration it replaces the mutated source code with the original files.
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
            print("Running coverage on: " + suite_file)

            # If the first round, calculate the coverage for the test suite.
            # Run coverage on the initial un-mutated files.
            try:
                # Run the coverage process on the test suite
                subprocess.check_output(["coverage","run","--source",source_folder,"-m","py.test", suite_file])
            except subprocess.CalledProcessError as e:
                # Failure encountered
                # print(e.output.decode("utf-8"))
                pass
            # Get the output of the coverage process
            coverage_output = subprocess.check_output(["coverage","report"])
            
            print("-----Coverage report------")
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
            
            # Benchmark results to be saved and compared against when performing
            # tests on the mutated code.
            bench_results = parse_test_results(test_result)
            bench_failed = bench_results[0]
            bench_passed = bench_results[1]
            
            print("Benchmark failed tests: " + str(bench_failed))
            print("Benchmark passed tests: " + str(bench_passed))
            
            # Dictionary to store results for each test suite
            result = {}
            # Add results to results dictionary
            result['coverage'] = coverage
            result['num_tests'] = num_tests
            result['bench_failed'] = bench_failed
            result['bench_passed'] = bench_passed
            result['num_muts_caught'] = 0
            result['num_muts'] = 0
            
            # Record result in result dictionary
            record[suite_fname] = result

    # Do mutation and collect test results.
    else:
        mut_error = False
        print("----------Mutation round: " + str(c) + " -----------------")
        # This attempts to mutate the source code. Run_mutation returns a bool
        # where true if a mutation is performed.
        mutation_performed = run_mutation(source_folder)
        print("Mutation performed: " + str(mutation_performed))
        # Run each test suite we have created
        for suite_fname in test_list:
            suite_file = test_path + suite_fname
            # Increment the count of mutations if one is performed.
            if mutation_performed:
                record[suite_fname]['num_muts'] = record[suite_fname]['num_muts'] + 1
            
            print("Running tests on: " + suite_file)
            # Run tests using subprocess module.
            try:
                test_result = subprocess.check_output(["py.test","--tb=line",suite_fname],cwd='tests')
            except subprocess.CalledProcessError as e:
                test_result = e.output.decode("utf-8")
            
            # Parse the results for the mutated source code.
            mutated_results = parse_test_results(test_result)
            
            print("Parsed results on mutated code: " + str(mutated_results))
            # Get the number of failures on the mutated code for this test suite
            mutated_failed = mutated_results[0]
            # Print test result if failed_no is 0. This is for debugging purposes.
            # If number of mutations failed is 0, it's likely an error has occured.
            if (mutated_failed == 0) and (not mut_error) :
                print("--------------TEST RESULTS:---------------" + str(test_result))
                mut_error = True
            
            print("***Mutation failures vs initial failures***")
            print("Mutated failures: " + str(mutated_failed) + ", Initial failures: " + str(record[suite_fname]['bench_failed']))
            
            # A mutant is killed if the number of failures for the test suite on the mutated source code
            # is higher than the benchmark level, OR if there is an error in one of the tests.
            if ((mutated_failed > record[suite_fname]['bench_failed']) or (mutated_results[2] > 0)):
                # Mutant caught - increment the mutant caught count
                record[suite_fname]['num_muts_caught'] = record[suite_fname]['num_muts_caught'] + 1
                print("Mutant caught! Mutant count: " + str(record[suite_fname]['num_muts_caught']))
                
    c += 1
    # Copy source code back to remove mutations.
    file_copy.restore_from_temp(source_path)

# Print the record dictionary.
print(record)
# Save the record to a .csv file
save_output(record)