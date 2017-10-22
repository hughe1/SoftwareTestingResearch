# Creates new test files based on a test suite contained in a directory
# Written for Software testing and reliability 2017
# By Hugh Edwards & Annie Zhou
# 15/10/2017

# Usage:
#   provide the existing test file directory as argument
#	Set parameters
#	Run the file
#	New test classes will be generated in a directory called "new_tests"

import os
import sys
import random
import re

################################################################################
# Parameters
################################################################################

#number of new test classes to create
NUMBER_SUITES = 50

#number of classes in each test suite
NUMBER_CLASSES = 30

################################################################################

# get path to the tests directory of the project from command line argument
path = sys.argv[1]
directory = os.fsencode(path)
proj_name = os.path.basename(os.path.dirname(path))
# directory = os.getcwd()

headers = []
all_classes = []

# loop through all files starting/ending in a certain string
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.startswith("test_") and filename.endswith(".py"):

        file_origin_suite = open(path + "/" + filename)
        string_origin_suite = file_origin_suite.read()

        # regex is used to break up the classes/methods
        # so that things like "data_class=" don't get mistaken
        # for a class

        # attempt to break into classes first, so nested methods stay together
        # TODO: methods in the same class are not split up and will be included
        # in their entirety - consider if should be broken up?
        tokens = re.split((r'(\nclass\W)'), string_origin_suite)

        if len(tokens) == 1:
            # split by class didn't work, try split by methods
            # prioritise splitting by lines startin with "@" as these are
            # "@pytest.mark...", "@mock..." marker lines that MUST stay
            # in front of the class/method for the test to run correctly
            # see https://docs.pytest.org/en/latest/mark.html
            tokens = re.split((r'(\n@|\ndef\W)'), string_origin_suite)

        # keep track of the imports that precedes the classes/methods
        if not tokens[0].startswith(("\ndef","\nclass","\n@")):
            # attempt to break into imports part and method part (if any)
            start = re.split((r'(\n@|\ndef\W)'), tokens[0])
            if len(start) == 1:
                headers.append(tokens[0])
                tokens = tokens[1:]
            else:
                # there are some other methods following the import section
                headers.append(start[0])
                tokens = start[1:] + tokens[1:] #put before rest

        # merge the "def" and "class" (split terms) with subsequent term
        classes = []
        i = 0
        while i < len(tokens):
            if tokens[i].startswith("\n@"):
                # this is a pytest marker line, need to combine with
                # subsequent method
                tokens[i+3] = ''.join(tokens[i:i+4])
                i += 3

            if tokens[i].startswith(("\ndef","\nclass")):
                # this is a class or method keyword, need to combine with
                # content of the class/method
                classes.append(tokens[i] + tokens[i+1])
                i += 1
            else:
                classes.append(tokens[i])
            i += 1

        for a_class in classes:
            all_classes.append(a_class)

        print("Processed: " + filename + "\n")
    else:
        continue

    num_suites = 0
    while num_suites < NUMBER_SUITES:

        picked_classes = [] #reset classes to empty
        num_classes = 0

        while num_classes < NUMBER_CLASSES:

            #create a new random test suite of 6 classes
            picked_classes.append(random.choice(all_classes))
            num_classes += 1

        #create a file with containing that test suite
        fname = "test_suite_" + str(num_suites) + ".py"

        # create folder to store test suites if needed
        if not os.path.exists(proj_name):
            os.makedirs(proj_name)

        fout = open(proj_name  + "/" + fname, 'w')

        #write in the header portion (imports)
        for a_header in headers:
            fout.write(a_header)

        for a_class in picked_classes:
            fout.write(a_class)

        num_suites += 1
