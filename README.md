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