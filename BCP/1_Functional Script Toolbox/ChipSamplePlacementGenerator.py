# Chip Sample Placement Generator
# Accepts 14 x 38 pixel image files and converts black pixels
# to pillar/well locations on chip. Generates an Axysys-compatible
# .txt file which can be used to specify printing locations.

# - Changelog -
# Created on 03/05/2014 by BCP.
# Updated on 03/10/2014

# - Pseudocode -
# Accept 14 x 38 pixel image file as input OR output error if wrong dimensions
# Threshold image file to 1 bit depth (0 for black, 1 for white)
# Convert ON/OFF image data to position data relative to pillar/well A01's absolute position
# Compile data into .txt file(s) with appropriate space delimiters for export to Axysys

from scipy import misc
from PIL import *
import Tkinter, tkFileDialog, os, subprocess, numpy
from operator import add

# - Functions -
# Allows user to select an image file using system dialog
def pickAFile(**options):
    path = tkFileDialog.askopenfilename()
    return path
        
# Allows user to select directory using system dialog    
def pickAFolder(**options):
	folder = tkFileDialog.askdirectory()
	if folder == '':
		folder = os.path.dirname()
	return folder

# Function writes data to text files
def writeTextDoc(fileName,whatToWrite):
    with open(fileName + ".txt", 'w') as f: # Open for writing
        # Write text to file, stripping brackets from list
        # To strip commas and brackets see below
        #f.write(str(whatToWrite).strip("[],"))
        f.write(str(whatToWrite).strip("[]"))
        f.close() # Closes the file

def printDebug(item):
    print item
    print len(item)

# - Load image -
# Image selected using system dialog    
image = pickAFile()

# - Visual tool -
# Remove comment below to show the image selected in a new window
# showimage = subprocess.call(image, shell=True)

# - Open image, threshold, check size, & store pixel data
# Image is opened as an Image object
image = Image.open(image)

# Convert (threshold) image to black/white (0/1)
image = image.convert("1")

# Check that the image is the appropriate size:
# 14 x 38 pixels 
if image.size != (14,38):
    print "Error: Image dimensions must be 14 x 48 pixels!"

# Stores the pixel data in a list
# 0 means black, 255 means white
data = list(image.getdata())

# - Create masks -
# Row mask    
rowMask = list(xrange(14))*38

#Column mask
columnMask = sorted(list(xrange(38))*14)

# - Specify chip parameters -
# Center-to-center distance between pillars/wells is 1.5 mm +/- 5 microns
increment = 1.5 #mm

# The x-position in mm of pillar/well A01
xA01 = 217.5 #mm

# The y-position in mm of pillar/well A01
yA01 = 58.5 #mm

# - Give masks physical meaning -
# The x-position of each column for all rows is the same
rowOffset = list((x*increment+xA01 for x in rowMask))

# The y-position of each row for all columns is the same
columnOffset = list((y*increment+yA01 for y in columnMask))

# - Apply masks to data -
# Sums the 0 (black) or 255 (white) list with the mask position list
# All elements selected will keep their original values from the mask
# All elements not selected will be offset by 255
xPositions = map(add, rowOffset, data)
yPositions = map(add, columnOffset, data)

# - Filter -
# Keeps only the elements of the Positions lists which have
# the same value as the matching elements in the masks
xPrint = list((x for x in xPositions if x in rowOffset))

yPrint = list((y for y in yPositions if y in columnOffset))

# - Verify -
# Used to tally up the total number of sites selected
printTally = list((x for x in data if x == 0))

# Compares the output list lengths to the total number of sites
# selected for printing, these numbers should match
if len(printTally) != len(xPrint) or len(printTally) != len(yPrint):
    print "ERROR: Printing locations are not accurate. Debug code!"
    
# - Write to File -
writeTextDoc("xCoordinates", xPrint) 
writeTextDoc("yCoordinates", yPrint)

# - Debugging tools -
##printDebug(data)
##printDebug(rowMask)
##printDebug(columnMask)
##printDebug(rowOffset)
##printDebug(columnOffset)
##printDebug(xPositions)
##printDebug(yPositions)
##printDebug(xPrint)
##printDebug(yPrint)
##printDebug(printTally)
