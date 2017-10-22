# Mutates python code by changing arithmetic and logical operators.
# Written for Software testing and reliability 2017
# By Matt Perrott
# 19/10/2017

# Usage:
#  

from os import listdir
from os.path import isfile, join
import os
import shutil
import random

###############################################################################

# Set of arithmetic operators
arith_set = set(["<=","<",">=",">","=="])
# Set of logical operators
logic_set = set(["or","and"])
# List of all operators to choose from
all_ops = list(arith_set | logic_set)
# List of arithemtic and logical sets
set_list = [arith_set, logic_set]

###############################################################################

# Given an operator, get a random mutation
def get_mutation(op):
    for s in set_list:
        if op in s:
            # Choose randomly from the set difference of the operator and set it
            # came from
            op_after = random.choice(list(s - set([op])))
            return op_after
    # If a mutation can't be found just return the original operator
    return op

# If the before operator exists in a list of lines, change a random op_before 
# into a op_after and return a new file.
# file_lines -> the list of lines to check from the read file
# new_file -> the new file that is being written
def mutate_file(file_lines, new_file):
    found_indexes = []
    # Shuffle the operator list
    random.shuffle(all_ops)
    # If an op exists in the file, set it as the operator to mutate
    for op in all_ops:
        # Get a valid mutation based on op
        op_after = get_mutation(op)
        # if op in list(logic_set):

        # Add spaces to ensure only op is picked up
        # e.g. so "and" isn't mutated in the word "band"
        op_before = " " + op + " "
        op_after = " " + op_after + " "

        # else:
        #     op_before = op
        # Check how many op_before operators exist in file, append the indexes
        # of lines with an operator in them to a list
        for i in range(0, len(file_lines)):
            if op_before in file_lines[i]:
                found_indexes.append(i)
        # If len > 0, we know we found that operator in list and can stop
        if len(found_indexes) > 0:
            break
    # Get mutated operation based on op chosen
    # op_after = get_mutation(op_before)
    print("Mutation: " + op_before + "   to   " + op_after) 
    # Catch the case that a mutation is not performed
    if op_after == op_before:
        return -1
    # If an operator was found, choose a random index of the line that will
    # be mutated
    if len(found_indexes) > 0:
        # Choose random index to mutate
        index_m = random.choice(found_indexes)
        # For each line in the file, write a new line the same as before, except
        # where the index is the line to be mutated. Replace the op_before with
        # op_afer on this line
        for i in range(0,len(file_lines)):
            if i == index_m:
                new_line = file_lines[i].replace(op_before,op_after, 1)
                new_file.write(new_line)
            else:
                new_file.write(file_lines[i])
        # Return the new file that has been created with a mutation
        return new_file
    # If no operator in this file are found, return error
    return -1

# Run the mutation.
# NOTE: This mutates the source code passed in!!! Don't call this if not
# using source control!
def run_mutation(source_folder):
    source_path = source_folder + "/"
    file_list = [f for f in listdir(source_path) if (isfile(join(source_path, f)) and (not f.startswith('_')) and f.endswith('.py'))]
    # Keep record for if a mutation has occurred in iteration
    mutated = False
    # Same source files, not to be mutated
    src = source_folder
    # Keep a counter of how many files have been tried
    file_length = len(file_list)
    # TODO: This loop will stop once the counter reaches the number of files 
    # that we have tried to mutate. It doesn't check that all files have been
    # tried, however. This should be implemented.
    j = 0
    while j < (file_length / 4):
        j += 1
        # Pick a random file to mutate
        i = random.randint(0, file_length-1)
        
        # print("File list: " + str(file_list))
        # print("File: " + str(file_list[i]))
        old_file = open(src + "/" + file_list[i])
        
        # Store old file as list of lines
        lines = old_file.readlines()
        # print(lines)
        # Open a new file that can be written to
        new_file = open(src + "/" + file_list[i], 'w')
        print("FILE LIST[i]: " + file_list[i])
        try:
            # Try mutating the new file (2 times)
            for l in range(0,2):
                new_file = mutate_file(lines, new_file)
            new_file.close()
            mutated = True
            print("Mutation on " + file_list[i] + " successful.")
            # If a file mutation is successful, move to next test suite
            # break
        except Exception, e:
            print("Exception: " + str(e))
            # If the file cannot be mutated, try more files
            continue
        
    
    if not mutated:
        print("No mutation applied for " + str(src))

    return mutated
 
if __name__ == "__main__":
    run_mutation()