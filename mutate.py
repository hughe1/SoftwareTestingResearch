from os import listdir
from os.path import isfile, join
import shutil
import random

# Folder where the code to mutate lives
source_folder = "test_source"
# Path where code to mutate lives
initial_path = source_folder + "/"

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
    print("MUTATE FILE")
    print("file_lines: " + str(file_lines))
    print("new_file: " + str(new_file))
    found_indexes = []
    # Check how many op_before operators exist in file, append the indexes
    # of lines with an operator in them to a list
    for i in range(0, len(file_lines)):
        print("i: " + str(i))
        if op_before in file_lines[i]:
            found_indexes.append(i)
    print("Found indexes: " + str(found_indexes))
    # If an operator was found, choose a random index of the line that will
    # be mutated
    print("Length found_indexes: " + str(len(found_indexes)))
    if len(found_indexes) > 0:
        print("HAZAAAA")
        # Choose random index to mutate
        index_m = random.choice(found_indexes)
        print("index_m: " + str(index_m))
        print("WOOPAAAA")
        # For each line in the file, write a new line the same as before, except
        # where the index is the line to be mutated. Replace the op_before with
        # op_afer on this line
        for i in range(0,len(file_lines)):
            print("i: " + str(i))
            if i == index_m:
                new_line = file_lines[i].replace(op_before,op_after)
                new_file.write(new_line)
            else:
                new_file.write(file_lines[i])
        # Return the new file that has been created with a mutation
        print("new_file: " + str(new_file))
        return new_file
    # If no operator in this file are found, return error
    return -1
        

# c - number of mutations to perform, and hence new sets of files
c = 0
while c < 3:
    print("c: " + str(c))
    print("--------------")
    # Same source files, not to be mutated
    src = source_folder
    # Append iteration number to destination files
    dest = source_folder + str(c+1)
    # Copy source files to new destination so they can be mutated
    shutil.copytree(src, dest)
    c += 1
    j = 0
    file_length = len(file_list)
    while j < file_length:
        i = random.randint(0, file_length-1)
        print("random i: " + str(i))
        j += 1
        old_file = open(dest + "/" + file_list[i])
        lines = old_file.readlines()
        print("lines: " + str(lines))
        new_file = open(dest + "/" + file_list[i], 'w')
        print("file: " + str(new_file))
        try:
            new_file = mutate_file(lines, new_file)
            new_file.close()
            break
        except:
            continue
    
    