from os import listdir
from os.path import isfile, join
import os
import shutil
import random

# Folder where the code to mutate lives
source_folder = "test_source"
# Path where code to mutate lives
initial_path = source_folder + "/"
# Ouput path
output_path = "output/"

# Set of arithmetic operators
arith_set = set(["<=","<",">=",">","=="])
# Set of logical operators
logic_set = set(["or","and","not"])
# List of all operators to choose from
all_ops = list(arith_set | logic_set)
# List of arithemtic and logical sets
set_list = [arith_set, logic_set]
# List of files in the folder of code to mutate
file_list = [f for f in listdir(initial_path) if isfile(join(initial_path, f))]

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
# new_file -> the file that is being written
def mutate_file(file_lines, new_file):
    found_indexes = []
    # Shuffle the operator list
    random.shuffle(all_ops)
    # If an op exists in the file, set it as the operator to mutate
    for op in all_ops:
        if op in list(logic_set):
            op_before = " " + op + " "
        else:
            op_before = op
        # Check how many op_before operators exist in file, append the indexes
        # of lines with an operator in them to a list
        for i in range(0, len(file_lines)):
            if op_before in file_lines[i]:
                found_indexes.append(i)
        if len(found_indexes) > 0:
            break
    # Get mutated operation based on op chosen
    op_after = get_mutation(op_before)
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

# Copy from_path to to_path even if it already exists
def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


#################### MAIN LOOP TO PRODUCE NEW TEST SUITES ####################
def run_mutation(number_mutations):
    mutation_count = 0
    # c - number of mutations to perform, and hence new sets of files
    c = 0
    while c < number_mutations:
        # Keep record for if a mutation has occurred in iteration
        mutated = False
        # Same source files, not to be mutated
        src = source_folder
        # Append iteration number to destination files
        dest = output_path + source_folder + str(c+1)
        # Copy source files to new destination so they can be mutated
        copy_and_overwrite(src, dest)
        # shutil.copytree(src, dest)
        c += 1
        # Keep a counter of how many files have been tried
        j = 0
        file_length = len(file_list)
        # 
        # TODO: This loop will stop once the counter reaches the number of files 
        # that we have tried to mutate. It doesn't check that all files have been
        # tried, however. This should be implemented.
        while j < file_length:
            # Pick a random file to mutate
            i = random.randint(0, file_length-1)
            j += 1
            old_file = open(dest + "/" + file_list[i])
            # Store old file as list of lines
            lines = old_file.readlines()
            # Open a new file that can be written to
            new_file = open(dest + "/" + file_list[i], 'w')
            try:
                # Try mutating the new file
                new_file = mutate_file(lines, new_file)
                new_file.close()
                mutation_count += 1
                mutated = True
                # If a file mutation is successful, move to next test suite
                break
            except:
                # If the file cannot be mutated, try more files
                continue
        # If it reaches this point, a mutation hasn't occurred
        if not mutated:
            print("No mutation applied for " + str(dest))

    print("MUTATIONS PERFORMED: " + str(mutation_count))
    return mutation_count
 
if __name__ == "__main__":
    run_mutation(10)