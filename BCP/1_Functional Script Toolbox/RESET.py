# Reset.py
"""
Automates resetting blank example files for testing 
sorting scripts.
"""

# - Log -
# Created 06/27/2014 by BCP.

# - Import -
import os, shutil

# - Code -

# Get current working directory
path = os.getcwd()

# Delete the TEST folder if it exists
if os.path.isdir(path + '/TEST') == True:
	shutil.rmtree(path + '/TEST')

# Change directory to the desired plate
os.chdir('20140311_Plate_3276')

# Copy the desired time points and all items contained within
shutil.copytree('TimePoint_1', path + '/TEST' + '/TimePoint_1')
shutil.copytree('TimePoint_2', path + '/TEST' + '/TimePoint_2')

# Change directory
os.chdir(path)

# Copy the desired test sorting script into TEST
shutil.copy('DirtyZSort.py', path + '/TEST')

print 'Complete!'