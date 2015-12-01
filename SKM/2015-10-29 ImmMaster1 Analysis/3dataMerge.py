# Montage the images.
import numpy as np
from glob import glob
import os
from time import time
import glob
import re
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import scipy.misc
from skimage import exposure

def to8(arg1):
	return (((arg1/65536.)*256).astype(int))

# Get the available image directories.
dirs = glob.glob('images\*')

imRE = re.compile('.*\\\\(d\d-.*)')

# Make subfolders for raw images and store them.
for dir in dirs:
	imageList = glob.glob(dir + '\*')
	rawDir = dir + '\\raw'
	if rawDir not in glob.glob(dir + '\*'):
		os.mkdir(rawDir)
		for im in imageList:
			os.rename(im, rawDir + '\\' + re.search(imRE, im).group(1))

# Get the minimum intensity values from IgG for each 2D/3D, each time point, each channel.
dims = ['2D', '3D']
times = ['d0', 'd2', 'd5']
channels = ['c1', 'c2', 'c3']
minRE = re.compile('images\\\\(.*)')
#condis = [re.search(minRE, con).group(1) for con in glob.glob('images\*') if 'IgG' in con]

# Make list of all IgG images.
#imList = []
#for condi in condis:
#	imList = imList + glob.glob('images\\' + condi + '\\raw\\*')

# Get the mins for the IgG.
#mins = {}
#for dim in dims:
#	for time in times:
#		for channel in channels:
#			#mins[dim + '-' + time + '-' + channel] = 1000.
#			tempMin = [imName for imName in imList if ((dim in imName) and (time in imName) and (channel in imName))]
#			cuts = []
#			if ('c1' in imName):
#				for imName in tempMin:
#					cuts.append(np.percentile(plt.imread(imName), 5))
#			else:
#				for imName in tempMin:
#					cuts.append(np.percentile(plt.imread(imName), 90))
#			mins[dim + '-' + time + '-' + channel] = np.mean(cuts)
#			print np.mean(cuts)

# Get the max intensity values for the stains. Apply the lowest max to the IgG.
#imList = []
#condis = [re.search(minRE, con).group(1) for con in glob.glob('images\*') if (('IgG' not in con) and ('PBS' not in con))]
#for condi in condis:
#	imList = imList + glob.glob('images\\' + condi + '\\raw\\*')			
#stainRE = re.compile('.*-.*-(.*)')
#stains = list(set([re.search(stainRE, x).group(1) for x in condis]))

#imList = []
#for condi in condis:
#	imList = imList + glob.glob('images\\' + condi + '\\raw\\*')

#maxes = {}
#for dim in dims:
#	for time in times:
#		for stain in stains:
#			for channel in channels:
#				#maxes[dim + '-' + time + '-' + stain + '-' + channel] = 3000.
#				tempMax = [imName for imName in imList if ((dim in imName) and (time in imName) and (stain in imName) and (channel in imName))]
#				tempMaxes = []
#				for imName in tempMax:
#					tempMaxes.append(np.percentile(plt.imread(imName), 95))
#				maxes[dim + '-' + time + '-' + stain + '-' + channel] = np.mean(tempMaxes)
#				print np.mean(tempMaxes)

# Get the max intensity for the controls based on the stains.
#for dim in dims:
#	for time in times:
#		for channel in channels:
#			#maxes[dim + '-' + time + '-' + 'Control' + '-' + channel] = 3000.
#			tempMax = [maxes[key] for key in maxes.keys() if ((dim in key) and (time in key) and (channel in key))]
#			maxes[dim + '-' + time + '-' + 'Control' + '-' + channel] = np.mean(tempMax)
#			print np.mean(tempMax)

# Make subfolders for the merges and generate them.
mergeRE = re.compile('.*-(.*)\\\\raw\\\\(d\d)-(\d\d)-(c\d).*')

for dir in dirs:
	print dir
	imageList = glob.glob(dir + '\\raw\\*')
	mergeDir = dir + '\\merge'
	if mergeDir not in glob.glob(dir + '\*'):
		os.mkdir(mergeDir)
		for im in imageList:
			# Get the image information.
			stain, day, num, channel = re.search(mergeRE, im).groups()
			if (channel == 'c1'):
				# Load the images.
				imName1 = im
				imName2 = im[:-6] + 'c2.tif'
				imName3 = im[:-6] + 'c3.tif'
				im1 = plt.imread(imName1)
				im2 = plt.imread(imName2)
				im3 = plt.imread(imName3)

				# Get the thresholds.
				#threshes = []
				#if (stain == 'IgG' or stain == 'PBS'): 
				#	threshes = [[mins[dim + '-' + day + '-' + 'c1'], maxes[dim + '-' + day + '-' + 'Control' + '-' + 'c1']],
				#	[mins[dim + '-' + day + '-' + 'c2'], maxes[dim + '-' + day + '-' + 'Control' + '-' + 'c2']],
				#	[mins[dim + '-' + day + '-' + 'c3'], maxes[dim + '-' + day + '-' + 'Control' + '-' + 'c3']]]
				#else:
				#	threshes = [[mins[dim + '-' + day + '-' + 'c1'], maxes[dim + '-' + day + '-' + stain + '-' + 'c1']],
				#	[mins[dim + '-' + day + '-' + 'c2'], maxes[dim + '-' + day + '-' + stain + '-' + 'c2']],
				#	[mins[dim + '-' + day + '-' + 'c3'], maxes[dim + '-' + day + '-' + stain + '-' + 'c3']]]

				# Apply the thresholds.
				#threshed = []
				#for idx, img in enumerate([im1, im2, im3]):
					# Remove to the min.
				#	img[np.where(img < threshes[idx][0])] = 0
					
					# Remove to the max.
				#	img[np.where(img > threshes[idx][1])] = threshes[idx][1]-1
					
					# Slide to the min.
				#	img[np.where(img > 0)] -= threshes[idx][0]
					
					# Expand to the max.
				#	scalingFactor = 1 / ((threshes[idx][1] - threshes[idx][0]) / 65535)
				#	img *= scalingFactor
					
				#	threshed.append(img)
					
				# Convert back to PIL.
				#im1 = to8(threshed[0]).astype('uint8')
				#im2 = to8(threshed[1]).astype('uint8')
				#im3 = to8(threshed[2]).astype('uint8')

				# Convert to 8 bit
				im1show = to8(im1).astype('uint8')
				im2show = to8(im2).astype('uint8')
				im3show = to8(im3).astype('uint8')

				# Get the min and max values for each channel.
				bmin, bmax, gmin, gmax, rmin, rmax = (np.min(im1show), np.max(im1show), np.min(im2show), np.max(im2show), np.min(im3show), np.max(im3show))

				# Rescale the intensities
				im1show = exposure.rescale_intensity(im1show)
				im2show = exposure.rescale_intensity(im2show)
				im3show = exposure.rescale_intensity(im3show)

				# Merge as RGB.
				#rgb = Image.fromarray(np.dstack((im1, im2, im3)), 'RGB')
				rgb = Image.fromarray(np.dstack((im1show, im2show, im3show)), 'RGB')

				# Label the image with the original min / max values for reference.
				outImage = ImageDraw.Draw(rgb)
				font = ImageFont.truetype('arial.ttf', 36)
				outImage.text((5,0), dir + '-' + day + '-' + num, (255, 255, 255), font = font)
				outImage.text((5,38), str(bmin) + '-' + str(bmax), (100, 100, 255), font = font)
				outImage.text((5,76), str(gmin) + '-' + str(gmax), (0, 255, 0), font = font)
				outImage.text((5,114), str(rmin) + '-' + str(rmax), (255, 0, 0), font = font)
				
				# Save as merge.
				rgb.save(mergeDir + '\\' + day + '-' + num + '.jpg', format='JPEG')
				
# Before proceeding with this, need to know how to level the images.
# Min from IgG. Max from max of true condition.
############

# Get the list of unique site names.
#wellNames = list(set([x[:-7] for x in imageNames]))
#
#for wellName in wellNames:
	# Generate the image names.
#	imageSubset = [wellName + 's' + str(x).zfill(2) + '.png' for x in range(1,17)]

	# Get the height and width of the image.
#	sampleImage = Image.open(imageSubset[0])
#	width = sampleImage.size[0]
#	height = sampleImage.size[1]
	
	# Generate the montage.
#	montage = Image.new('L', (width*4, height*4))
#	xCount = 0
#	yCount = 0
#	for imageName in imageSubset:
#		montage.paste(Image.open(imageName), (xCount * width, yCount * height))
#		if(xCount < 3):
#			xCount += 1
#		else:
#			xCount = 0
#			yCount += 1
			
#	montage.save(outDirectory + wellName[10:] + '.png')
#	montage.close()