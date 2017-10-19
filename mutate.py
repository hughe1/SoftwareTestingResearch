from os import listdir
from os.path import isfile, join
import os
import shutil
import random

NUMBER_MUTATIONS = 10

# Folder where the code to mutate lives
source_folder = "test_source"
# Path where code to mutate lives
initial_path = source_folder + "/"
# Ouput path
output_path = "output/"

# TODO: turn into a list of operators, not just single hardcoded ones
# Operator to mutate
op_before = "+"
# Operator to mutate to
op_after = "-"

# List of files in the folder of code to mutate
file_list = [f for f in listdir(initial_path) if isfile(join(initial_path, f))]

# If the before operator exists in a list of lines, change a random op_before 
# into a op_after and return a new file.
# file_lines -> the list of lines to check from the read file
# new_file -> the file that is being written
def mutate_file(file_lines, new_file):
    found_indexes = []
    # Check how many op_before operators exist in file, append the indexes
    # of lines with an operator in them to a list
    for i in range(0, len(file_lines)):
        if op_before in file_lines[i]:
            found_indexes.append(i)
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
                new_line = file_lines[i].replace(op_before,op_after)
                new_file.write(new_line)
            else:
                new_file.write(file_lines[i])
        # Return the new file that has been created with a mutation
        return new_file
    # If no operator in this file are found, return error
    return -1

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)
        

# c - number of mutations to perform, and hence new sets of files
c = 0
while c < NUMBER_MUTATIONS:
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
            mutated = True
            # If a file mutation is successful, move to next test suite
            break
        except:
            # If the file cannot be mutated, try more files
            continue
    # If it reaches this point, a mutation hasn't occurred
    if not mutated:
        print("No mutation applied for " + str(dest))
    
    