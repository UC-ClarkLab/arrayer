# BackgroundSubtract.py
"""
TEST script for removal of background signal from fluorescence
microscope images using the rolling ball method.
"""

# - Change Log -
# Created 07/21/2014 by BCP.

# - Image Processing Objectives -

# - Determine image quality
# - Add scale bars
# - Flat field correction
# - Background subtraction (rolling ball method)
# - "Scrapping" size filter of artifacts
# - Dual channel (or multi channel) image segmentation with Hoescht and TL - make sure to save segmentation masks!
# - Measure average intensity and area of identified cells
# - Are more advanced morphometric measurements applicable?
# - Data visualization - heatmap of signal intensity/well on chip - charts indicating percentage of cells with co-staining for other signals

# - Pseudocode - 

# Import image
# Subtract background using rolling ball method
# Save as new image
# Measure image quality of new and original images
# Compare quality of images

# - Import required modules -
import os, shutil, re
#from scipy import ndimage
from PIL import Image 
from PIL import ImageFilter

import numpy as np
#import cv2

# - Function Definitions -



# - Code Body -

# Obtains the current working directory of the folder.
directory = os.getcwd()

file = '20140629-BCP-Falcon-96-10x-7dayUndirDiffIII_A10_s003_w1.TIF'
#image = ndimage.imread(file)
image = Image.open(file)
imageL = image.convert('L;16')



# cap = cv2.VideoCapture('vtest.avi')

# fgbg = cv2.createBackgroundSubtractorMOG2()

# while(1):
    # ret, frame = cap.read()

    # fgmask = fgbg.apply(frame)

    # cv2.imshow('frame',fgmask)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
        # break

# cap.release()
# cv2.destroyAllWindows()

try:
	image0 = image.filter(ImageFilter.BLUR)  
	#image1 = image.filter(ImageFilter.CONTOUR)  
	#image2 = image.filter(ImageFilter.DETAIL)  
	#image3 = image.filter(ImageFilter.EDGE_ENHANCE)  
	#image4 = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  
	#image5 = image.filter(ImageFilter.EMBOSS)  
	#image6 = image.filter(ImageFilter.FIND_EDGES)  
	#image7 = image.filter(ImageFilter.SMOOTH)  
	#image8 = image.filter(ImageFilter.SMOOTH_MORE)  
	#image9 = image.filter(ImageFilter.SHARPEN)  
except:
	print 'Image has wrong mode.'

#array = np.array(image.convert('L'))
#width, height = image.size
image.show()

print 'Image:', image.size, image.mode
print 'ImageL:', imageL.size, imageL.mode





	






#fgbg = cv2.BackgroundSubtractorMOG()

#background = image.BackgroundSubtractor.getBackgroundImage()
#background.show()
#foreground = Image.fromarray(fgbg.apply(array))
#foreground.show()