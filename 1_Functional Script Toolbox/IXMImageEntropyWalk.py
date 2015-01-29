#IXMImageEntropyWalk.py
"""
Place into plate folder. Run. Measures entropy of all images and exports
data to a .csv document.
"""

# Measure image quality through image entropy
# Bigger entropy means more noise/liveliness/color/business
# Usually an image with bigger entropy is more complex and "useful"
# Adapted from:
#http://brainacle.com/calculating-image-entropy-with-python-how-and-why.html

# - Change Log -
# Created 07/23/2014 by BCP.
# Edited 07/25/2014 by BCP to enable batch processing.

# - Function Definitions-

def image_entropy(imagename):
	"""Calculates the Shannon Entropy of an image."""
	from PIL import Image
	import math
	import collections
	import os
	
	path, name = os.path.split(imagename)
	image = Image.open(imagename)
	histogram = image.histogram()
	histogram_length = sum(histogram)
	
	samples_probability = [float(h) / histogram_length for h in histogram]

	entropy = -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
	
	header = ['File', 'Shannon_Entropy']
	data = [name, entropy]
	
	return header, data
	
def write_to_csv(iterableitem, header, filename):
	"""Writes data to a comma-separated value file (.csv)."""
	import csv	
	import os
	
	path = os.getcwd()
	
	try:
		iterator = iter(iterableitem)
				
		with open(filename + '.csv', 'wb') as csvfile:
			headwriter = csv.writer(csvfile)
			headwriter.writerow(header)
			
			if type(iterableitem) is dict:
				writer = csv.writer(csvfile)
				for key, value in iterableitem.items():
					writer.writerow([key, value])
			
			else:
				writer = csv.writer(csvfile)
				writer.writerows(iterableitem)
		print 'Writing data to ' + path + '\\' + filename + '.csv ...'
		
	except TypeError:
		print 'Input', iterableitem, 'is not iterable.'

# - Body -

import math
import matplotlib.pyplot as plt
import os, shutil, re

# Obtains the current working directory of the folder.
directory = os.getcwd()

# Make a list of all subdirectories (time point folders)
subdirectories = os.walk(directory).next()[1]

# Initialize output list
shannon_entropy = []

# Iterate through the subdirectories
# Identify metadata in the file name/directory
# If a relevant file is found, caclulate the entropy
# Move on until finished
for subdirectory in subdirectories:
	
	# Identify the time point metadata from the file directory
	try:
		timepoint = re.search('TimePoint_\d{1,2}', subdirectory)
		timepoint = timepoint.group(0)
	except: 
		print 'No timepoints identified.'
		
	imagefolders = os.walk(subdirectory).next()[1]
	#print imagefolders
	imagefiles = []	
	
	for folder in imagefolders:
		files = os.listdir(directory + '\\' + subdirectory + '\\' + folder)
				
		# In that list of files, iterate file-by-file and take the relevant actions
		for image in files:
	
			image = str(image)			
			print image
			# Search file name for thumbnail metadata
			thumbcheck = re.search('Thumb', image)
			
			# Search file name for relevant image metadata
			wavecheck = re.search('w\d', image)
			extcheck = re.search('.TIF', image)
			
			imagepath = str(directory + '\\' + subdirectory + '\\' + folder + '/' + image)
			#print imagepath
			
		
			# If file is a thumbnail, do nothing
			if (thumbcheck):
				print 'Skipping thumb...'
				pass
			elif (wavecheck and extcheck):
				header, data = image_entropy(imagepath)
				shannon_entropy.append(data)
				print 'Measuring image entropy...'
							
				# # Plot histogram of data
				# histogram = plt.hist(pixel_intensities, bins=100, normed=True, histtype='bar')
				# plt.xlabel('12-bit Depth Signal Intensity')
				# plt.ylabel('Normalized Frequency (%)')
				# plt.title(name)
				# plt.savefig(name, format = png, transparent = True)
			else:
				pass		
						
	# Write to file	
	write_to_csv(shannon_entropy, header, 'Entropy_Measurements')
	print 'Complete!'

	
"""
METHOD VALIDATION
"""	
# img = '20140629-BCP-Falcon-96-10x-7dayUndirDiffIII_A10_s003_w1.TIF'
# shannon_entropy = image_entropy(img)
# print shannon_entropy
# write_to_csv(shannon_entropy, 'TEST_CSV')
# from PIL import Image
# import numpy as np
# name = str(img)
# image = Image.open(name)
# pixel_intensities = list(image.getdata())
# # Plot histogram of data
# histogram = plt.hist(pixel_intensities, bins=100, normed=True, histtype='bar')
# plt.xlabel('12-bit Depth Signal Intensity')
# plt.ylabel('Normalized Frequency (%)')
# plt.title('Histogram of Image Intensity')
# plt.show()

