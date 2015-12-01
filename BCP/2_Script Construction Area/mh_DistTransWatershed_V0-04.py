# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 12:40:46 2015

@author: Brian Perea
"""

# The distance transform is often combined with the watershed for segmentation.
# Here is an example (which is available with the source in the mahotas/demos/
# directory as nuclear_distance_watershed.py).

import mahotas
# from os import path
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt

import skimage
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, closing



try:
    # nuclear_path = '20140805_A01_s004_w1_TimePoint_1.TIF'
    # nuclear_base = mahotas.imread(nuclear_path)
    nuclear_base = (np.load('CLAHE_pickle')*65535).astype(np.uint16)
except NameError:
    print('Error: no such file found.')
    # nuclear_path = path.join('data', 'nuclear.png')

# Make a separate .py file for setting up batch image processing
# Define image processing code as a function and call from MAIN
"""
import os
# Obtains the current working directory of the folder.
directory = os.getcwd()

DAPI = os.listdir(directory + '/w1')
FITC = os.listdir(directory + '/w2')
TexasRed = os.listdir(directory + '/w3')
TL = os.listdir(directory + '/w4')

## Make a list of all subdirectories (time point folders)
#subdirectories = os.walk('.').next()[1]
#
#for subdirectory in subdirectories:
#
#    # Convert the subdirectory name to string type
#    subname = str(subdirectory)
#
#	# Identify channel (wavelength) metadata
#	waveexpression = re.compile('w\d')
#	# Search file name string for site metadata
#	wavecheck = re.search(waveexpression, subname)
#
#    if wavecheck:
#        files = os.listdir(directory + '/' + subname)
#
"""
# The code is not very complex. Start by loading the image
# and preprocessing it with a Gaussian blur:


print(nuclear_base.shape)
nuclear = nuclear_base[:,:]
nuclear = mahotas.gaussian_filter(nuclear, 1.)

# Equalizing the histogram reveals illumination variations across well
#illumination = exposure.equalize_hist(nuclear_base)
#background = nuclear.astype(np.uint16)
#background1 = opening(background, disk(10)) # was disk(10)
#background2 = closing(background, disk(10))
#test_correct = exposure.rescale_intensity(nuclear.astype(np.uint16) - background.astype(np.uint16))
#test_correct1 = nuclear.astype(np.uint16) - background1.astype(np.uint16)
#
#wakka = nuclear.astype(np.uint16)
#
##check_illum = exposure.equalize_hist(test_correct)
#
##plt.imshow(test_correct1, cmap=plt.cm.gray)
#
#p2, p98 = np.percentile(nuclear, (2, 98))
#nuclear = exposure.rescale_intensity(nuclear, in_range=(p2, p98))



t_otsu = threshold_otsu(nuclear)
print(t_otsu)
threshed  = (nuclear > t_otsu)



# Find ROI = Regions Of Interest
ROI = np.zeros_like(nuclear_base).astype(np.uint16)
np.copyto(ROI, nuclear_base, where=threshed)
ROI = ROI.astype(np.uint16)

local_otsu = skimage.filter.rank.otsu(ROI, disk(15), mask=threshed)
local_threshed = (ROI > local_otsu)

# None of the filters below help much
#eq_adapthist = exposure.equalize_adapthist(alt_mask)
#eq_hist = exposure.equalize_hist(alt_mask)
#rescale = exposure.rescale_intensity(alt_mask)

# Investigate clusters of cells
colonies, n_colonies  = mahotas.label(threshed)
sizes = mahotas.labeled.labeled_size(colonies)
# print(sizes)

too_small = np.where(sizes < 100)
colonies = mahotas.labeled.remove_regions(colonies, too_small)
#colonies = mahotas.labeled.remove_bordering(colonies)
colonies, n_colonies = mahotas.labeled.relabel(colonies)
print('Found {} colonies.'.format(n_colonies))
# plt.imshow(colonies)
# print(colonies)


# Investigate nuclei within cell clusters

# Now, we compute the distance transform:
distances = mahotas.stretch(mahotas.distance(local_threshed))

# We find and label the regional maxima:
Bc = np.ones((9,9))

maxima = mahotas.morph.regmax(distances, Bc=Bc)
spots,n_spots = mahotas.label(maxima, Bc=Bc)
print('Found {} maxima.'.format(n_spots))
# plt.imshow(spots)

# Finally, to obtain the image above, we invert the distance transform
# (because of the way that cwatershed is defined) and compute the watershed:
surface = (distances.max() - distances)
areas = mahotas.cwatershed(surface, spots)
areas *= local_threshed

labeled, n_nucleus  = mahotas.label(local_threshed)


sizes = mahotas.labeled.labeled_size(labeled)
# print(sizes)
too_small = np.where(sizes < 100)
labeled = mahotas.labeled.remove_regions(labeled, too_small)
# plt.imshow(labeled)
labeled, n_nucleus = mahotas.labeled.relabel(labeled)
print('Found {} nuclei.'.format(n_nucleus))

areas = labeled

# We used a random colormap with a black background for the final image.
# This is achieved by:
import random
from matplotlib import colors as c
colors = map(plt.cm.jet,range(0, 256, 4))
random.shuffle(colors)
colors[0] = (0.,0.,0.,1.)
rmap = c.ListedColormap(colors)
# plt.imshow(areas, cmap=rmap)

# Plotting
fig1, axs = plt.subplots(2, 3)
axs[0,0].imshow(nuclear, cmap=plt.cm.gray)
axs[0,1].imshow(threshed, cmap=plt.cm.gray)
axs[0,2].imshow(ROI, cmap=plt.cm.gray)
axs[1,0].imshow(local_threshed, cmap=plt.cm.gray)
axs[1,1].imshow(colonies, cmap=plt.cm.gray)
axs[1,2].imshow(areas, cmap=rmap)
plt.show()