# nWavelengthIXMSort.py
"""
Place into plate folder. Run. Sorts any number of IXM time points from any type of IXM data set into folders by channel ONLY. Also labels the time point at which each file was acquired.
"""

# - Change Log -
# Created 06/27/2014 by BCP.
# Updated 07/03/2014 by BCP to handle sorting by well and site number.
# Forked 07/10/2014 by BCP from SortIXMTimepoints.py to sort IXM data by channel only.

# - Import required modules -
import os, shutil, re

# - Pseudocode - 

# Iterate through the subdirectories
# Identify metadata in the file name/directory
# Create the relevant folder hierarchy for sorting
# Rename and move the file in one step

# - Code -

# Obtains the current working directory of the folder.
directory = os.getcwd()

# Make a list of all subdirectories (time point folders)
subdirectories = os.walk('.').next()[1]

# Iterate through the subdirectories
# Identify metadata in the file name/directory
# Create the relevant folder hierarchy for sorting
# Rename and move the file in one step
for tree in subdirectories:
	
	# Identify the time point metadata from the file directory
	timepoint = re.search('TimePoint_\d{1,3}', tree)
	timepoint = timepoint.group(0)

	# Create a list of files in the current subdirectory
	files = os.listdir(directory + '/' + timepoint)
		
	# In that list of files, iterate file-by-file and take the relevant actions
	for file in files:
		
		# Split file name from the extension 				
		name, ext = os.path.splitext(file)
		#print name
		
		# Old file name (including full path)
		old = directory + '/' + timepoint + '/' + name + ext
		
		# Check for thumbnails
		# Identify thumbnail metadata
		thumb = re.compile('Thumb')
		# Search file name for thumbnail metadata
		thumbcheck = re.search(thumb, name)
		
		# If file is a thumbnail, do nothing
		if (thumbcheck):
			print 'Marking thumbnail for removal...'
			pass
		else:
					
			# Convert the file name to string type	
			filename = str(file)
			
			# Identify channel (wavelength) metadata
			waveexpression = re.compile('w\d')
			# Search file name string for site metadata
			wavecheck = re.search(waveexpression,filename)
			
			# If well metadata is identified:
			if (wavecheck):
				
				# Provide status indicators for the user
				print 'Image identified!'
				# Indicate which aspects of the image metadata have been identified and what their values are
				print 'Channel: ' + wavecheck.group()
							
				# If the relevant site folder does not exist, create it
				if os.path.isdir(directory + '/' + wavecheck.group()) == False:
					os.makedirs(directory + '/' + wavecheck.group())
													
				# Simultaneously move and rename the file by modifying its full path
				os.rename(old, directory + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext)
				# Print output file location
				print directory + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext
												
	# Clean up	
	shutil.rmtree((directory + '/' + timepoint))
	print 'Complete!'

