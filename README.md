# SWEN90006 Research Project 2017

### Hypothesis 

> Branch coverage is more effective in finding seeded faults in
> open source software programs than number of test cases.

### Process

1. We found a three publically available, medium sized Python programs, each having an included test suite containing at least 100 test cases.
2. We then split up the each test suite randomly into 50 sets of test cases (resulting in 150 ‘sample test suites’ overall)
	- Each test suite contains a random number of test cases from 1-100.
	- Test cases need not be non-overlapping in this step.
3. We then measured and recorded, for each set of test cases:
	- Branch Coverage; using coverage.py, and
	- The number of test cases in the suite
4. We then used a python script to introduce 100 mutations to the program. A new version of each program was saved with a single new bug each, so that there are 10 altered version of each program, each with one manually entered ‘simulated’ bug.
	- Mutations were purely logical and arithmetic.
5. We then ran each set of 50 test cases against each of the mutated programs, recording whether each test case set detects the fault; ie, if any test case fails.


### Todo

1. Finish `mutate.py`
2. Write a driver script. It should do the following:
	i) Runs `mutate.py` to generate the mutants for a program
		- This creates a folder containing multiple test suite files (test_suite_01.py, test_suite_02.py, ...)
	ii) Runs `create_suites.py` to create the test suites for a program
		- This creates a folder of containing multiple mutated program directories (mutant_program_01, mutant_program_02)
		- Each of these folders contains a duplicate of the entire program and all its files, with one mutation
	iii) For each test suite file:
		a) Calculates the branch coverage of the test suite on the ORIGINAL, unmutated program (using coverage.py)
		b) Calculates the number of test cases in the test suite file (naively, count number of times "def test_" occurs in the file)
		c) Runs the test suite on every single mutated program, recording whether all tests pass, or if one or more tests fail (if one or more test cases fail, the mutant is caught)
	iv) Saves a file `results.csv` with the following headers
			- Test suite # (ie test_suite_04)
			- Coverage
			- Number of test cases
			- Number of mutants caught
			- Total number of mutants
			- Mutant score (caught/total mutants)
		Each row corresponds to a single test suite file

   
### Mutation

`mutate.py` is the start of our mutator.

It's not currently working, but the idea is it looks at all the files in
the specified directory, and looks for a specific operator to mutate.
Once mutated, it saves a new directory. It keeps working through the files
trying to pick random operators to mutate.

### Creation of test suites

`create_suites.py` is the start of our test splitter.

It looks for classes in the specified test suite and grabs them, creating new 
test suites.