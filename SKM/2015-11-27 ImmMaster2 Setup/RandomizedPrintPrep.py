# Protocol for 5 constant media conditions over 5 days.
# Each of the 5 constant condition is fixed and stained
# at the end with each of 5 different 1' antibody comobos.
# 2' is the same across all (Hoescht, Ms Green, Rb Red).

from random import randint
from copy import deepcopy
import csv
from decimal import Decimal

# Media key.
# 1 is mTeSR, 2 is APEL, 3 is NeuroDiff, 4 is MesoDiff, 5 is EndoDiff.
mediaKey = {'1':'mTeSR', '2':'APEL', '3':'NeuroDiff', '4':'MesoDiff', '5':'EndoDiff', '6':'DualSmad'}

# Stain key.
# 1 is Ms nestin, 2 is Rb Pax6, 3 is Rb Brach, 4 is Rb GATA4, 5 is IgG.
stainKey = {'1':'MsNesRbOct4', '2':'MsOct4RbPax6', '3':'MsOct4RbBra', '4':'MsOct4RbGATA4', '5':'MsGATA4RbBra', '6':'MsNesRbPax6', '7':'MsGATA4RbGATA4', '8':'IgG', '9':'PBS'}

# Specify experiment parameters.
CHIPoriginX = 217.5
CHIPoriginY = 58.7
CHIPoffsetX = 27.65
CHIPoffsetY = 0.01
NUMcontrols = 6 # Number of controls. In this case,
#				  they are assumed to be constant.
NUMagents = 0 # Number of agents.
NUMdoses = 0 # of concentrations of agents.
NUMstains = 8 # of stains that will be used. (Does not count the PBS blanks for extras.)
NUMchips = 3
WELLS = 532

# Function to convert a position value (0-531 for chip 1, 532-1061 for chip 2, etc.)
# to an (x,y) value.
def orderToXY(val):
	row = val / 14
	col = val % 14
	xPos = round(CHIPoriginX + (col * 1.5) + (row / 38 * CHIPoffsetX), 2)
	yPos = round(CHIPoriginY - (row % 38 * 1.5) - (row / 38 * CHIPoffsetY), 2)

	#print 'row : ' + str(row)
	#print 'col : ' + str(col)
	#print 'xPos : ' + str(xPos)
	#print 'yPos : ' + str(yPos)

	return (xPos, yPos)

# Derive number of replicates, number of extras.
REPS = WELLS / (NUMcontrols * NUMstains)
EXTRA = WELLS % (NUMcontrols * NUMstains)

### Generate the fully enumerated list of conditions.
printList = []

for i in range(1,NUMcontrols+1):
	for j in range(1,NUMstains+1):
		for r in range(REPS):
			printList.append(str(i)+'-'+str(j))

# Add on the extras.
for e in range(EXTRA):
	printList.append('1-9') # In this case, 1-9 is mTeSR with "PBS" staining.

### Generate pseudo-random chip layouts. (can use truerandom.getnum() if necessary.)
# See OLD version for truerandom usage.
chipList = []
for chip in range(NUMchips):
	# Fisher Yates shuffle.
	swap = ''
	printListCopy = deepcopy(printList)

	for i in reversed(range(1,len(printListCopy))):
		rand = randint(0, i)
		print str(i) + ' - ' + str(rand) # Enable to watch progress.
		swap = printListCopy[i]
		printListCopy[i] = printListCopy[rand]
		printListCopy[rand] = swap

	chipList.append(printListCopy)

# Output the condition maps to csv's.
for chipNum, chip in enumerate(chipList):
	with open('chip' + str(chipNum) + '.csv', 'wb') as outFile:
		wr = csv.writer(outFile, delimiter = ',', quoting = csv.QUOTE_NONE)
		for x in range(38):
			wr.writerow(chip[0+14*x:14+14*x])

### Generate the media print locations for 3 chips.
mediaLocs = {x : [] for x in mediaKey.keys()}
for chipOffset, chip in enumerate(chipList):
	for index, condi in enumerate(chip):
		mediaLocs[condi[0]].append(orderToXY(index + chipOffset*532))

# Export the positions to a csv for 3 chips.
with open('media3chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('media3chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for media in sorted(mediaLocs.keys()):
			wrX.writerow([pos[0] for pos in mediaLocs[media]])
			wrY.writerow([pos[1] for pos in mediaLocs[media]])

# Export the positions to a csv for 2 chips.
with open('media2chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('media2chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for media in sorted(mediaLocs.keys()):
			wrX.writerow([pos[0] for pos in mediaLocs[media]][:len(mediaLocs[media])/3*2])
			wrY.writerow([pos[1] for pos in mediaLocs[media]][:len(mediaLocs[media])/3*2])

# Export the positions to a csv for 1 chip.
with open('media1chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('media1chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for media in sorted(mediaLocs.keys()):
			wrX.writerow([pos[0] for pos in mediaLocs[media]][:len(mediaLocs[media])/3])
			wrY.writerow([pos[1] for pos in mediaLocs[media]][:len(mediaLocs[media])/3])

### Generate the antibody print maps for each of the 3
### timepoints of 1 chip each (D2, D5, D7).
stainLocs = {x : [] for x in stainKey.keys()}
for chipOffset, chip in enumerate(chipList):
	for index, condi in enumerate(chip):
		stainLocs[condi[2]].append(orderToXY(index + chipOffset*532))

# Export the positions to a csv for pairs of chips.
with open('stainChipXD2.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('stainChipYD2.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for stain in sorted(stainLocs.keys()):
			wrX.writerow([pos[0] for pos in stainLocs[stain][len(stainLocs[stain])/3*2:]])
			wrY.writerow([pos[1] for pos in stainLocs[stain][len(stainLocs[stain])/3*2:]])

with open('stainChipXD5.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('stainChipYD5.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for stain in sorted(stainLocs.keys()):
			wrX.writerow([pos[0] for pos in stainLocs[stain][len(stainLocs[stain])/3:len(stainLocs[stain])/3*2]])
			wrY.writerow([pos[1] for pos in stainLocs[stain][len(stainLocs[stain])/3:len(stainLocs[stain])/3*2]])

with open('stainChipXD7.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('stainChipYD7.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for stain in sorted(stainLocs.keys()):
			wrX.writerow([pos[0] for pos in stainLocs[stain][:len(stainLocs[stain])/3]])
			wrY.writerow([pos[1] for pos in stainLocs[stain][:len(stainLocs[stain])/3]])
