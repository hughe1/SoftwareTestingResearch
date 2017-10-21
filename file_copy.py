from distutils.dir_util import copy_tree
from shutil import rmtree, copytree
import sys, os

# Ignore pattern to avoid copy sub-directories (avoids permission errors
# when copying lib folder, for example)
def ig_folder(dir, files):
    return [f for f in files if os.path.isdir(os.path.join(dir, f))]

# Copies all files in the src_path to a temporary folder in the current
# direction while ignoring all sub-directories
# src_path need to refer to a directory
def copy_to_temp(src_path):
    dir_name = os.path.basename(os.path.dirname(src_path))
    temp_name = dir_name + "_temp"

    if os.path.exists(temp_name):
        rmtree(temp_name) # remove first to avoid file exist error

    copytree(src_path, temp_name, ignore=ig_folder)

# Copies all files in the temporary folder in the current
# direction back to the src_path
# src_path need to refer to a directory
def restore_from_temp(src_path):
    dir_name = os.path.basename(os.path.dirname(src_path))
    temp_name = dir_name + "_temp"

    # copy_tree used to allow for merging into existing folders (with sub-directories)
    copy_tree(temp_name, src_path)
    # rmtree(temp_name)


# # Test stuff
# src_path = sys.argv[1]
# copy_to_temp(src_path)
# restore_from_temp(src_path)
