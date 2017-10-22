# SWEN90006 Research Project 2017

By Annie Zhou, Grace Johnson, Hugo Edwards and Matt Perrott.

### Hypothesis 

> Number of Test Cases is a more accurate measure of a test suite’s 
> effectiveness in finding seeded program faults than Statement Coverage.

### Process

1. We found two publically available, medium sized Python programs, each having 
   an included test suite containing at least 100 test cases.
2. We then split up each test suite randomly into 50 sets of test cases 
   (resulting in 100 ‘sample test suites’ overall) using `divide_test_suite.py`
  	- Each test suite contains a random number of test cases from 1-100.
  	- Test cases need not be non-overlapping in this step.
  	- The test suites required some manual editing to resolve some import errors
3. We then measured and recorded, for each set of test cases:
	- Statement Coverage; using coverage.py, and
	- The number of test cases in the suite
	- The initial number of passing and failing tests on the un-mutated source 
	  code
4. We then used a python script (`mutate.py`) to introduce mutations to the 
   program. Each time we introduced mutations, we saved the source code and ran
   each test suite over the mutated code to see if any of the tests picked up
   the mutation ("killing the mutant").
   - Mutations were purely logical and arithmetic.
5. The results for each test suite were updated after every test, and we
   repeated this process for a set number of mutations. This is driven by the 
   program `run_tests.py`.

### Usage

To use this program (generally):

1. Find an open source python program with existing test cases
2. `git clone` the repository to your local machine
3. Perform the necessary steps to get the tests running for that open source
   program. This step is crucial as it sets up the relevant paths and is needed
   to prevent import errors. This will require running a virtual env, and
   probably running a `make` or `tox` program (but depends on the open program).
4. Run `divide_test_suite.py` with the existing test folder as an argument
5. Replace the existing test files with the new files you have created. This 
   may require resolving some import errors.
6. Set the appropriate parameters in `run_tests.py`.
7. Do `python run_tests.py` and wait for the process to finish
8. Analyse your results in the .csv file created

**Warning**: This program mutates the files in the specified folder in 
`run_tests.py`. Don't run this code in a folder with files that aren't under
source control.

*Note*: Mutation analysis is computationally expensive, and this code has not
been written with efficiency as a priority. If you want to get results quickly,
choose small test suites and a low number of mutation rounds to perform.

### Note

The process of creating random test suites from other suites can be tedious
due to import errors. We have provided the test suites for the two programs
that we tested - *httpie* and *requests*. These are located in the folders
`httpie_tests` and `requests_tests`. Running these should produce similar
results to what we got, but due to the random nature of our mutations there
will be some differences. Also note that to get 50 test suites running over
50 mutation rounds took many hours of computation (doing less mutation rounds
is recommended).