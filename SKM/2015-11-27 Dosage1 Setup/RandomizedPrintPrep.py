# Protocol for 34 agents, 5 controls (mTeSR, APEL, Neuro, Meso, EndoDiff). 
# APEL will be used as the base to which each agent is added.
# (VPA left out for inclusion in teratogen screen)
# Each of the 34 agents is at 5 doses (literature ED50 with two logs to either side).
# Staining will be L/D, Oct4 + (Nestin or Pax6 ... depends on Immuno Master outcome),
# Oct 4 + Brachyury, Oct4 + GATA4. Each of these will require its own chip.
# 2' is the same across all (Hoescht, Ms Green, Rb Red).
# 9 chips total. 1 for D0 L/D + Immunostaining (1/4 of chip each).
# 4 for D2. 4 for D5.

from random import randint
from copy import deepcopy
import csv
from os import mkdir

# Control key.
controlKey = {'C1':'APEL', 'C2':'mTeSR', 'C3':'NeuroDiff', 'C4':'MesoDiff', 'C5':'EndoDiff'}

# Agent key. ****** Should add the middle dose value here in a tuple with the name e.g. in uM, but as a number. ******
agentKey = {'01':'CHIR99021', '02':'iCRT5', '03':'SB431542', '04':'LDN193189', '05':'DAPT', '06':'PD173074', '07':'OAC-1', '08':'Prostaglandin E2', '09':'SP600125', '10':'SB202190',
	 '11':'Go6983', '12':'Rosiglitazone', '13':'Activin A', '14':'TGF-B1', '15':'BMP4', '16':'FGF2', '17':'EGF', '18':'y-27632', '19':'Purmorpamine', '20':'Cyclopamine',
	 '21':'PD0325901', '22':'Indolactam', '23':'PS-48', '24':'Ly294002', '25':'Pifithrin-mu', '26':'Pyrintegrin', '27':'Sinomenine', '28':'ID-8', '29':'AICAR', '30':'IDE2',
	 '31':'Trichostatin A', '32':'5-Azacytidine', '33':'RepSox', '34':'3-Deazaneplanocin A'}

# Stain key. Live dead isn't included here.
stainKey = {'1':'Ecto', '2':'Meso', '3':'Endo', '4':'IgG'}

# Diluent key.
diluentKey = {'1': 'APEL'}

# Dose key. Represents the log modifier (e.g. -2 is 1/100).
doseKey = {'1':-2, '2':-1, '3':0, '4':1, '5':2}

# Specify experiment parameters.
CHIPoriginX = 217.5
CHIPoriginY = 58.7
CHIPoffsetX = 27.65
CHIPoffsetY = 0.01
NUMcontrols = 5 # Number of controls. In this case,
#				  they are assumed to be constant.
NUMagents = 34 # Number of agents.
NUMdoses = 5 # of concentrations of agents.
NUMstains = 4 # of stains that will be used. (Does not count the PBS blanks for extras.)
NUMchips = 9
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

### Generate the fully enumerated list of conditions.
printList = []

# Agents with doses first.
for i in sorted(agentKey.keys()):
	for j in sorted(doseKey.keys()):
		for r in range(3): # The number of replicates per agent.
			printList.append(i + '-' + j)

# Add on the controls.
for c in sorted(controlKey.keys()):
	for r in range(4): # The number of control replicates.
		printList.append(c)

# Add on the extras.
for r in range(2): # The number of remaing empty spots.
	printList.append('C1') # APEL is chosen as the 'extra'

### Generate pseudo-random chip layouts. (can use truerandom.getnum() if necessary.)
# See OLD version for truerandom usage.
chipList = []
for chip in range(NUMchips):
	# Fisher Yates shuffle.
	swap = ''
	printListCopy = deepcopy(printList)

	for i in reversed(range(1,len(printListCopy))):
		rand = randint(0, i)
		# print str(i) + ' - ' + str(rand) # Enable to watch progress.
		swap = printListCopy[i]
		printListCopy[i] = printListCopy[rand]
		printListCopy[rand] = swap

	chipList.append(printListCopy)

# Output the condition maps to csv's.
mkdir('arrayerFiles')
for chipNum, chip in enumerate(chipList):
	with open('arrayerFiles/chip' + str(chipNum) + '.csv', 'wb') as outFile:
		wr = csv.writer(outFile, delimiter = ',', quoting = csv.QUOTE_NONE)
		for x in range(38):
			wr.writerow(chip[0+14*x:14+14*x])

########################################
### Generate the positional information.
########################################
### APEL diluent (720 nl) prints.
diluentLocs = {x : [] for x in diluentKey.keys()}
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		# In the future where more than one diluent is used, this will need to be changed.
		if ('-2' in condi or '-4' in condi):
			diluentLocs['1'].append(orderToXY(index+chipOffset*532))
# This is 204 wells per chip, so would need 1 well of APEL per chip.

# Export the position to a csv for 8 chips.
with open('arrayerFiles/diluent8chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('arrayerFiles/diluent8chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for diluent in sorted(diluentLocs.keys()):
			wrX.writerow([pos[0] for pos in diluentLocs[diluent]])
			wrY.writerow([pos[1] for pos in diluentLocs[diluent]])

# Export the position to a csv for 4 chips.
with open('arrayerFiles/diluent4chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('arrayerFiles/diluent4chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for diluent in sorted(diluentLocs.keys()):
			wrX.writerow([pos[0] for pos in diluentLocs[diluent]][:len(diluentLocs[diluent])/2])
			wrY.writerow([pos[1] for pos in diluentLocs[diluent]][:len(diluentLocs[diluent])/2])

### Agent prints.
# Will need *2* 96 well plates for this part, in which 17 3-well units (51 wells) are 
# dedicated on each to the agent dilutions. First get the low dilution and print
# it at 800. Then get the middle dilution and print it at 80 and 800. Then get the high
# dilution and print at 80 and 800. Move on to next agent. etc. Technically don't need
# to wash between dilution since it ascends, just eject buffer sheath fluid into wash basin.
agentLocs = {x : [] for x in printList if ('-1' in x or '-3' in x or '-5' in x)}
agentAmts = {x : [] for x in printList if ('-1' in x or '-3' in x or '-5' in x)}
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		if ('-1' in condi):
			agentLocs[condi].append(orderToXY(index + chipOffset*532))
			agentAmts[condi].append('0.8')
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		if ('-2' in condi):
			agentLocs[condi[:3] + '3'].append(orderToXY(index + chipOffset*532))
			agentAmts[condi[:3] + '3'].append('0.08')
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		if ('-3' in condi):
			agentLocs[condi].append(orderToXY(index + chipOffset*532))
			agentAmts[condi].append('0.8')
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		if ('-4' in condi):
			agentLocs[condi[:3] + '5'].append(orderToXY(index + chipOffset*532))
			agentAmts[condi[:3] + '5'].append('0.08')
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		if ('-5' in condi):
			agentLocs[condi].append(orderToXY(index + chipOffset*532))
			agentAmts[condi].append('0.8')
# 24 prints for the "-1", 48 for "-3" and "-5".
# agentAmts are left uncommented here, though now that we
# are sorting they aren't necessary (and aren't exported).

# Export the position to a csv for 8 chips.
with open('arrayerFiles/agent8chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('arrayerFiles/agent8chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for agent in sorted(agentLocs.keys()):
			wrX.writerow([pos[0] for pos in agentLocs[agent]])
			wrY.writerow([pos[1] for pos in agentLocs[agent]])

# Export the position to a csv for 4 chips.
with open('arrayerFiles/agent4chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('arrayerFiles/agent4chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for agent in sorted(agentLocs.keys()):
			wrX.writerow([pos[0] for pos in agentLocs[agent]][:len(agentLocs[agent])/2])
			wrY.writerow([pos[1] for pos in agentLocs[agent]][:len(agentLocs[agent])/2])

# Export the amounts to a csv for 8 chips.
#with open('arrayerFiles/agent8chipAmt.txt', 'wb') as outFileX:
#	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
#	for agent in sorted(agentAmts.keys()):
#		wrX.writerow([pos for pos in agentAmts[agent]])

# Export the amounts to a csv for 4 chips.
#with open('arrayerFiles/agent4chipAmt.txt', 'wb') as outFileX:
#	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
#	for agent in sorted(agentAmts.keys()):
#		wrX.writerow([pos for pos in agentAmts[agent]][len(agentAmts[agent])/2:])

### Control prints.
controlLocs = {x : [] for x in controlKey.keys()}
for chipOffset, chip in enumerate(chipList[1:]):
	for index, condi in enumerate(chip):
		# In the future where more than one diluent is used, this will need to be changed.
		if ('C' in condi):
			controlLocs[condi].append(orderToXY(index+chipOffset*532))

# Export the position to a csv for 8 chips.
with open('arrayerFiles/control8chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('arrayerFiles/control8chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for control in sorted(controlLocs.keys()):
			wrX.writerow([pos[0] for pos in controlLocs[control]])
			wrY.writerow([pos[1] for pos in controlLocs[control]])

# Export the position to a csv for 4 chips.
with open('arrayerFiles/control4chipX.txt', 'wb') as outFileX:
	wrX = csv.writer(outFileX, delimiter = ' ', quoting = csv.QUOTE_NONE)
	with open('arrayerFiles/control4chipY.txt', 'wb') as outFileY:
		wrY = csv.writer(outFileY, delimiter = ' ', quoting = csv.QUOTE_NONE)
		for control in sorted(controlLocs.keys()):
			wrX.writerow([pos[0] for pos in controlLocs[control]][:len(controlLocs[control])/2])
			wrY.writerow([pos[1] for pos in controlLocs[control]][:len(controlLocs[control])/2])


