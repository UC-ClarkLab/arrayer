# -*- coding: utf-8 -*-
"""
Created on Wed May 06 15:13:37 2015

@author: Brian
"""

# Notes:
# the y value needs to DECREASE from 57.0 max to 4.5 min
# the x value needs to INCREASE from 219.0 min to 235.5 max
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import csv
    import re
except ImportError:
    print "Problem importing required module."

# - Specify chip parameters -
# Center-to-center distance between pillars/wells is 1.5 mm +/- 5 microns
increment = 1.5 #mm

# The x-position in mm of pillar/well A01
xA01 = 217.5 #mm
xA02 = 219.0 #mm this is the first interior x position
# all x positions must INCREASE from xA02 to 235.5 MAX

# The y-position in mm of pillar/well A01
yA01 = 58.5 #mm
yA02 = 57.0 #mm this is the first interior y position
# all y positions must DECREASE from yA02 to 4.5 MIN

## The x-distance between chips
#xToNext = 27.65 #mm
#
## The y-distance between chips
#yToNext = -0.01 #mm

# - Set up print -

# Number of distinct conditions
conditions = range(0, 84, 1)

# Available pillars/wells on chip
available = np.zeros((36, 12))

# Full chip map filled with '*' for use in mapping conditions
# Each well is a mixture of 3 components (hence array of depth 3)
chip = np.full((38, 14, 3), '*', dtype='|S16')


# Row mask
rowMask = list(xrange(12))*36

#Column mask
columnMask = sorted(list(xrange(36))*12)

# The x-position of each column for all rows is the same
xgen = list((x*increment+xA02 for x in rowMask))

# The y-position of each row for all columns is the same
ygen = list((yA02-y*increment for y in columnMask))

idx = np.asarray(xgen).reshape(36, 12)
idy = np.asarray(ygen).reshape(36, 12)
#with open('constconditions.csv', 'r') as f:
#    reader = csv.reader(f)
#    for row in reader:
#        print row

# read conditions from Excel (converted to csv)
df = pd.DataFrame.from_csv('constconditions.csv', header=None, index_col=None).fillna(value="media")

# Build the chip layout
array = df.values
array = np.expand_dims(array, axis=0).reshape(7, 12, 3)
build_chip = array
for n in range(0, 4):
    build_chip = np.vstack((build_chip, array))
build_chip = np.resize(build_chip, (36, 12, 3))#.resize(38, 14, 3)
print build_chip.shape
#chip[1:37, 1:13, :] = build_chip
#array = chip
array = build_chip
print array.shape
# order of indices in array is [row, column, component]

# From a list of conditions containing duplicates, find the 
# unique condition names and map them to a dictionary
unq = np.unique(array) # iterate on this, call from dct dictionary!!
unq = unq[unq != '*']

# Generate the corresponding map of volumes to be printed
dct = dict()
for i in unq:

    dct[i] = np.sum(array == i, axis=2)

#    print dct[i]

    xposi = idx[dct[i] != 0] # only the important x positions
    yposi = idy[dct[i] != 0] # only the important y positions
    voli = (dct[i] * 0.266)[(dct[i] * 0.266) != 0] # volumes to be printed
#    print sum(voli)/1000

    print i
    print xposi.shape
    print yposi.shape
    print voli.shape

    name = re.sub(r'[^\w]', '-', i)
#    print name
    namegen = ['_xpos', '_ypos', '_vol']
    k = 0
    for j in [xposi, yposi, voli]:
        np.savetxt('{0}{1}.txt'.format(name, namegen[k]), j, fmt='%1.4f', newline=", ")
#        print 'Writing to file {0}{1}.txt'.format(name, namegen[k])
        k += 1
    plt.imsave('{0}.png'.format(name), dct[i])

