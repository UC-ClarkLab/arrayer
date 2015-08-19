#IXM_Montage-ConventionalXY.py
"""Place into plate folder sorted by wavelength. Run. Will create a montages of
each wavelength separately with files ordered according to the .HTD metadata file 
from the IXM which contain indicators of sites selected. Variables XWells and YWells must be specified manually!!"""

# - Change Log -
# Created 08/19/2014 by BCP.
# Updated 08/26/2014 by BCP, completed metadata functionality and corrected montage construction.
# Updated 08/29/2014 by BCP, implemented a very basic method for compiling montages for standard culture dishes with X and Y wells. Variables XWells and YWells must be specified manually!!

# - To Do -
# - Enable automatic identification of the number of X and Y wells selected and implement in variables XWells and YWells
# - Add a combined RGB overlay for w1, w2, and w3 montages
# - Enable filling of gaps for irregularly shaped montages

# - Import required modules -
import numpy as np

from PIL import Image

from skimage import data, exposure
import skimage.io as io
import skimage.transform as transform

import os, re, csv, shutil
import os.path

try:
	from PyQt4.QtCore import QString
except ImportError:
	QString = str
	print 'You need to download PyQt4 first! It is available at: http://www.riverbankcomputing.co.uk/software/pyqt/download'

# - Function Definitions - 

def montageIXM(plate_path, labelled= True, rescale_intensity=True):
	""""Place into plate folder sorted by wavelength. Run. Will create a montages of
	each wavelength separately and a combined RGB overlay with files ordered according
	to the .HTD metadata file from the IXM which contain indicators of sites selected. 
	Function is set to automatically rescale image intensity unless otherwise specified."""
	# Obtains the working directory of the targeted folder.
	directory = plate_path
	
	# Make a list of all subdirectories (time point folders)
	subdirectories = os.walk('.').next()[1]
	#print subdirectories
	
	# Search for relevant metadata file
	files_in_directory = os.listdir(directory)
	#print files_in_directory
	
	for file in files_in_directory:
		htd_check = re.search('.HTD', file)
		if htd_check:
			metadata_file = file
	if 'metadata_file' in locals():
		print 'Metadata file "' + metadata_file +'" identified. Attempting to read...'
	else:
		print 'Error: Metadata file with .HTD extension is missing.'
		return
	
	# Reads the metadata file to a dictionary consisting of line title
	# and value pairs.
	with open(metadata_file, 'r') as csv_metadata:
		csv_reader = csv.reader(csv_metadata)
		csv_dict = dict()
		
		# Clean up messy formatting of lines read from the proprietary 
		# metadata file. Removes unnecessary spaces and quotations to 
		# put information in readily accessible format.
		for line in csv_reader:
			key = line.pop(0)
			#print line
			line = [item.split(' ',1)[1] for item in line]
			#print line
			i = 0
			for item in line:
				line[i] = re.sub('"|"','', item)
				i += 1
			#print line
			csv_dict.update({key:line})
		key_list = csv_dict.keys()
		#print key_list
	print 'Metadata file read successfully.'
	
	# Determine shape of montage based on metadata
	print 'Determining the shape of your montage based on metadata...'
	def list2string2int(list):
		# Converts a list of a single string to an integer
		list = int(re.sub("'|'",'', str(list)).replace('[','').replace(']',''))
		return list

	# Determine max columns and rows for sites selected
	site_selection_list = []
	for key in key_list:
		#print key
		site_selectorcheck = re.search('SiteSelection\d{1,2}', key)
		if site_selectorcheck:
			print site_selectorcheck.group(), ' identified!'
			site_selection_list.append(csv_dict[key])
			
	#print site_selection_list
	col_selector = []
	row_selector = []
	for bool_list in site_selection_list:
		# Identify number of columns selected
		col_selection_counter = bool_list.count('TRUE')
		col_selector.append(col_selection_counter)
		
		# Identify number of rows selected
		index = 0
		for bool in bool_list:
			if bool == 'TRUE':
				row_selector.append(index)
			index += 1
			
	row_selector_counter = []
	for i in row_selector:
		row_selector_counter.append(row_selector.count(i))
	#print row_selector_counter
	
	columns = max(col_selector)
	rows =  max(row_selector_counter)
	print '(columns, rows):', (columns, rows)
	
	# Determine well setup for plate
	XWells = 9
	YWells = 2
	#print XWells, YWells
	grid_shape = (columns*XWells, rows*YWells)
	print 'Montage grid dimensions:', grid_shape
	
	print 'Searching for image wavelength folders...'
	
	for subdirectory in subdirectories:
		
		# Search file name string for site metadata
		wavecheck = re.search('w\d', subdirectory)
		
		if wavecheck:
			
			print 'Image folder ' + str(wavecheck.group()) + ' identified.'
			
			# Create a list of files in the current subdirectory
			files = os.listdir(directory + '/' + subdirectory)
			#print files
			files = [subdirectory + '/' + file for file in files]
			#print files
			
			total_files = len(files)
			print 'There are ' + str(total_files) + ' image files.'
				
	
	###################################################################################
	# METHOD 5 - NumPy binary files 
	# http://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html#numpy.save
	###################################################################################	
	
	# Issue: Works as well as (or better than!) any other method! Now add feature to correct how the montage is displayed
	# Add method to create RGB composite overlay montage of DAP, FITC, and TexasRed channels
	# Add text labelling functionality for verification using PIL.ImageDraw.Draw.text(xy,text,fill=None,font=None,anchor=None)?
	# e.g. https://code.google.com/p/esutil/source/browse/trunk/esutil/sqlite_util.py
	
			montage = wavecheck.group()
			# temp = directory + '/tmp'
			# if os.path.exists(temp):
				# shutil.rmtree(temp)
			# os.mkdir(temp)
			# Rescale intensity of images
			print '(XWells, YWells):', (XWells, YWells)
			sites_per_well = total_files/(XWells*YWells)
			print 'sites_per_well: ', sites_per_well
			
			for k in range(YWells):
				for i in range(XWells):
					row_id = 0 
					counter = 0
					new_row = True
					print 'Starting well: ', i
										
					for j in range(sites_per_well):
						# print 'counter: ', counter
						# print 'new_row: ', new_row
						image = Image.open(files[j+i*sites_per_well+k*sites_per_well*XWells])		
						#print 'File:', j+i*sites_per_well+k*sites_per_well*XWells			
						image = np.array(image, dtype=np.int16)
						#print 'Converted to NumPy array.'
						
						# To rescale the intensity of EACH image INDEPENDENTLY, uncomment the line below
						# and comment the other exposure.rescale_intensity(montage_np) after the for loop ends
						#image = exposure.rescale_intensity(image)
						
						image = transform.pyramid_reduce(image, downscale=10)
						#print 'Resizing image now.'
						
						if new_row:
							row_builder = image
							# print 'Row initialized.'
							new_row = False
							
						else:
							row_builder = np.hstack((row_builder, image))
						# print 'row_builder: ' + str(row_builder.shape)
							
						
						
						if counter == rows: # If we reached the end of a row
							# print '(counter, rows):', (counter, rows)
							if row_id == 0: # If the row is the first row
								montage_np = row_builder
								#print 'montage_np: '+ str(montage_np.shape)
								
							else: # If the row is any other row
								#print 'montage_np: '+ str(montage_np.shape)
								# print 'row_builder: ', row_builder.shape
								montage_np = np.vstack((montage_np,row_builder))
							counter = 0
							row_id += 1
							new_row = True
							print('Row %d complete!' % row_id)
							# print row_builder.size
							#test = Image.fromarray(montage_np)
							#test.save(temp+'/Montage_%s_%s.tif' % (montage, row_str))
							print 'Well Montage Dimensions:', montage_np.shape	
						else:	
							counter += 1
							
					if i == 0:
						x_stack = montage_np
					else:
						x_stack = np.hstack((x_stack, montage_np))
					print 'Montage Dimensions in X Direction:', x_stack.shape
					
				if k == 0:
					y_stack = x_stack
				else:
					y_stack = np.vstack((y_stack, x_stack))
				print 'Montage Dimensions in Y Direction:', y_stack.shape
				
			print('Compiling and rescaling the montage for viewing...')
			
			# To rescale the intensity of the TOTAL MONTAGE all at once, allowing visual comparison of image intensities,
			# leave the line below uncommented and comment the exposure.rescale_intensity(image) in the for loop, above
			well_stack = exposure.rescale_intensity(y_stack)
			
			img = Image.fromarray(well_stack)
			print('Montage %s created.' % montage)
			img.save('Montage_%s.tif' % montage)
			print('Montage %s saved!' % montage)
	
# - Main -

# Obtains the current working directory of the folder.
directory = os.getcwd()
montageIXM(directory)
print 'Complete!'