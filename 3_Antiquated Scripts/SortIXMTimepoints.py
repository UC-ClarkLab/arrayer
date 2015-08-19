# SortIXMTimepoints.py
"""
Place into plate folder. Run. Sorts any number of IXM time points from any type of IXM data set into folders by well, site number, and channel. Also labels the time point at which each file was acquired.
"""

# - Change Log -
# Created 06/27/2014 by BCP.
# Updated 07/03/2014 by BCP to handle sorting by well and site number.
# Updated 07/10/2014 by BCP to remove thumbs, sort by channel, and display more informative status indicators.

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
	timepoint = re.search('TimePoint_\d{1,2}', tree)
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
			
			# Identify well metadata
			wellexpression = re.compile('A\d\d')
			# Search file name string for well metadata
			wellcheck = re.search(wellexpression,filename)
			
			# Identify site metadata
			siteexpression = re.compile('s\d{1,4}')
			# Search file name string for site metadata
			sitecheck = re.search(siteexpression,filename)
			
			# Identify channel (wavelength) metadata
			waveexpression = re.compile('w\d')
			# Search file name string for site metadata
			wavecheck = re.search(waveexpression,filename)
			
			# If well metadata is identified:
			if (wellcheck):
				
				# Provide status indicators for the user
				print 'Image identified!'
				if (sitecheck):
					site = sitecheck.group()
				else:
					site = 'None'
				# Indicate which aspects of the image metadata have been identified and what their values are
				print 'Well: ' + wellcheck.group() + '\nSite: ' + site + '\nChannel: ' + wavecheck.group()
											
				# If site metadata is identified:
				if (sitecheck):
					#print 'Site identified!'
					print sitecheck.group()
					
					# If the relevant site folder does not exist, create it
					if os.path.isdir(directory + '/' + wellcheck.group() + '/' + sitecheck.group() + '/' + wavecheck.group()) == False:
						os.makedirs(directory + '/' + wellcheck.group() + '/' + sitecheck.group() + '/' + wavecheck.group())
														
					# Simultaneously move and rename the file by modifying its full path
					os.rename(old, directory + '/' + wellcheck.group() + '/' + sitecheck.group() + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext)
					# Print output file location
					print directory + '/' + wellcheck.group() + '/' + sitecheck.group() + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext
					
				# If well metadata AND no site metadata:				
				else:
					if os.path.isdir(directory + '/' + wellcheck.group() + '/' + wavecheck.group()) == False:
						os.makedirs(directory + '/' + wellcheck.group() + '/' + wavecheck.group())
											
					# Simultaneously move and rename the file by modifying its full path
					os.rename(old, directory + '/' + wellcheck.group() + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext)
					# Print output file location
					print directory + '/' + wellcheck.group() + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext
				
	# Clean up	
	shutil.rmtree((directory + '/' + timepoint))
	print 'Complete!'

