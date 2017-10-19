# SWEN90006 Research Project 2017

### Hypothesis 

> Branch coverage is more effective in finding seeded faults in
> open source software programs than number of test cases.

### Process

1. Pick 3 open source python programs with comprehensive test suites
2. For each program, split up the test suite in two seperate ways
    1. 50 sets of test cases of equal size
    2. 50 sets of test cases of differing size
3. Write a mutator script to make a single manual mutation in the source code
   and saving to a new directory each time (100 different directories, all with
   one mutation).
4. For each test suite in the first set, measure branch coverage of each test
   suite.
5. Run each test suite over the mutated code, measuring the mutants killed 
   (whether or not a test fails for each directory created).
6. Plot branch coverage vs effectiveness, and number of test cases vs 
   effectivness
   
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