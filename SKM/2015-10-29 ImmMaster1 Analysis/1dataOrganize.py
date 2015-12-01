# Run this script in parent directory for a given
# day to consolidate all image files into respective
# folders with naming in pattern:
# pP*rXXcYYwCsS.tif where p is plate name, r is row, c is col,
# w is wavelength, s is slice.

import glob
import os
import re

# If they don't already exist, create image and thumb root folders.
if 'thumbs' not in glob.glob('*'):
	os.mkdir('thumbs')

if 'images' not in glob.glob('*'):
	os.mkdir('images')
	
# Get the list of days in directory that
# haven't yet been sorted (based on whether
# or not there is a 'thumbs' directory for the
# thumbnail images.
plateList = [plate for plate in glob.glob('*_Plate_*') if not glob.glob(plate + '\\organized')]

# Get nested list of thumb images for each day from filtered plateList.
# Note that this doesn't handle multiple days where there are no thumbnails.
thumbList = [file for file in [glob.glob(plate + '\\Time*\\*Thumb.TIF') for plate in plateList]]

ImmInfoRE = re.compile('\d{8}-.*-(\d)D-D(\d).*\\\\.*\\\\.*_[A-Z](\d\d)_s(\d{1,3})_w(\d)')

# Move thumbs to respective day's thumbs folder,
# renaming to pP*rXXcYYwCsS.tif format.
for thumbs in thumbList:
	for thumb in thumbs:
		if ('TL' in thumb):
			vals = re.search(TLInfoRE, thumb)
			well = vals.group(3)
			site = vals.group(4)
			day = vals.group(2)

			newName = 'd' + day + 'w' + well + 's' + site.zfill(2) + 'c4.tif'
			os.rename(thumb, 'thumbs\\' + newName)
		elif ('Imm' in thumb):
			vals = re.search(ImmInfoRE, thumb)
			dim = vals.group(1)
			well = vals.group(3)
			site = vals.group(4)
			day = vals.group(2)
			channel = vals.group(5)

			newName =  dim + 'Dd' + day + 'w' + well + 's' + site.zfill(3) + 'c' + channel + '.tif'
			os.rename(thumb, 'thumbs\\' + newName)

# Now that the thumbnails are removed, get the actual images.
imageList = [file for file in [glob.glob(plate + '\\Time*\\*.TIF') for plate in plateList]]

for images in imageList:
	for image in images:
		dim, day, well, site, channel = re.search(ImmInfoRE, image).groups()
			
		if(well == '03' and int(site)%13 == 0):
			continue
		else:
			newName = dim + 'Dd' + day + 'w' + well + 'c' + channel + 's' + site.zfill(3) + '.tif'
			os.rename(image, 'images\\' + newName)
			
# To fix a messed up zfill of only 2.
#for imageName in imageList:
#	dim, day, well, site, channel = re.search(ImmInfoRE, imageName).groups()
#	newName = newName = dim + 'Dd' + day + 'w' + well + 'c' + channel + 's' + site.zfill(3) + '.tif'
#	os.rename(imageName, 'images\\' + newName)
	
# Create the an 'organized' folder so we know it is done for
# future reference.
for plate in plateList:
	os.mkdir(plate + '\\organized')