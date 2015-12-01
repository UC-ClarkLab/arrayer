#Zfill_Site.py
'''Place into plate folder. Run. Fills the site number with three digits to
facilitate sorting for the creation of montages and other image processing.'''

# - Change Log - 
# Created 08/26/2014 by BCP.

# - Import required modules -
import os, shutil, re

# - Function Definitions - 

# Obtains the current working directory of the folder.
directory = os.getcwd()

# Make a list of all subdirectories (time point folders)
subdirectories = os.walk('.').next()[1]

for subdirectory in subdirectories:
	
	# Search file name string for site metadata
	wavecheck = re.search('w\d', subdirectory)
	
	if wavecheck:
		
		# Create a list of files in the current subdirectory
		files = os.listdir(subdirectory)
		
		# In that list of files, iterate file-by-file and take the relevant actions
		i = 0
		for file in files:
			
			sitecheck = re.search('s\d{1,3}', file)
			
			if sitecheck:
				old = sitecheck.group()
				new = old[0] + old[1:].zfill(3)
				new_file = re.sub('s\d{1,3}', new, file)
				os.rename(directory +'\\'+ subdirectory +'\\'+ file, directory +'\\'+ subdirectory +'\\'+ new_file)
			i += 1
			print new_file
		files = os.listdir(subdirectory)
		#print files

print 'Complete!'