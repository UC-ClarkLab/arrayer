import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import glob
import re
import os

if 'plots' not in glob.glob('*'):
	os.mkdir('plots')

# Experimental info
dims = ['2D', '3D']
times = ['d0', 'd2', 'd5']
channels = ['c1', 'c2', 'c3']
mediaKey = {'1':'mTeSR', '2':'APEL', '3':'NeuroDiff', '4':'MesoDiff', '5':'EndoDiff'}
stainKey = {'1':'MsNes', '2':'RbPax6', '3':'RbBra', '4':'RbGATA4', '5':'IgG', '6':'PBS'}
dirs = glob.glob('images\*')

data = sqlite3.connect('DefaultDB.db')
d = data.cursor()

# Table names
# obtained with d.execute("SELECT name FROM sqlite_master WHERE type='table';")
# 'Per_c1ob'
# 'Per_c2ob'
# 'Per_c3ob'
# 'Per_Image'
# 'Experiment'
# 'sqlite_sequence'
# 'Experiment_Properties'
# 'Per_Experiment'
# 'Per_RelationshipTypes'
# 'Per_Relationships'
def getTableNames():
	d.execute("SELECT name FROM sqlite_master WHERE type='table';")
	return [str(x[0]) for x in d.fetchall()]
	
tableNames = getTableNames()

# Can get column names within table via:
# d.execute('select * from bar')
# cols = list(map(lambda x: x[0], d.description))
def getColNames(tableName):
	d.execute('SELECT * FROM ' + tableName)
	return [str(x[0]) for x in d.description]
	
colNames = {x : getColNames(x) for x in tableNames}

# For retrieving data from sql database without having to write sql.
def getData(cols, table):
	if type(cols) == list:
		colString = cols[0]
		for x in cols[1:]:
			colString += ', ' + x
		
	else:
		colString = cols
		
	return d.execute('SELECT ' + colString + ' FROM ' + table).fetchall()

# For plotting.
# Modified from http://matplotlib.org/examples/api/barchart_demo.html
titleRE = re.compile('images\\\\(.*)')
def plotMarkerPercs(title, xLabel, yLabel, legendLabels, d0vals, d0devs, d2vals, d2devs, d5vals, d5devs):
	title = re.search(titleRE, title).group(1)
	N = 3
	ind = np.arange(N)
	width = 0.2
	
	means = np.column_stack((d0vals, d2vals, d5vals))
	devs = np.column_stack((d0stds, d2stds, d5stds))
	devs[devs < 0] = 0
	
	#print means
	#print devs
	
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind+0.5*width, means[0], width, color = 'g', yerr = devs[0], ecolor='k')
	rects2 = ax.bar(ind+1.5*width, means[1], width, color = 'r', yerr = devs[1], ecolor='k')
	rects3 = ax.bar(ind+2.5*width, means[2], width, color = 'y', yerr = devs[2], ecolor='k')
	rects4 = ax.bar(ind+3.5*width, means[3], width, color = '0.75', yerr = devs[3], ecolor='k')
	
	ax.set_title(title)
	ax.set_xticks(ind+2.5*width)
	ax.set_xticklabels(xLabel)
	ax.set_ylabel(yLabel)
	
	ax.set_ylim(bottom=0)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	
	ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), (legendLabels), loc = 'center left', bbox_to_anchor=(1, 0.5))
	
	plt.savefig('plots\\' + title + '.png')
	
	#raw_input('Press enter to continue...')
	
	del fig
	del ax
	
	
print 'Declarations done.'

############################
### Nuclear stain counts ###
############################
# Need to get the image numbers to match with the directories.
imNumbers = {x[0] : str(x[1]) + '\\' + str(x[2]) for x in getData(['ImageNumber', 'Image_PathName_c1', 'Image_FileName_c1'], 'Per_Image')}
obCounts = getData(['ImageNumber', 'Image_Count_c1ob', 'Image_Count_c2ob', 'Image_Count_c3ob'], 'Per_Image')
obCounts = {x[0] : [x[1], x[2], x[3]] for x in obCounts if x[1] > 100}
c2rels = getData(['ImageNumber', 'c2ob_Number_Object_Number', 'c2ob_Children_c3ob_Count', 'c2ob_Parent_c3ob'], 'Per_c2ob')
c3rels = getData(['ImageNumber', 'c3ob_Number_Object_Number', 'c3ob_Children_c2ob_Count', 'c3ob_Parent_c2ob'], 'Per_c3ob')

print 'Data retreived.'

# Finds the number of co-stains to subtract for each.
c2relCounts = {}
c3relCounts = {}
for im in imNumbers.keys():
	print im
	c2byc3 = len([ob for ob in c2rels if(ob[0] == im and ob[2] > 0)])
	c3byc2 = len([ob for ob in c3rels if(ob[0] == im and ob[2] > 0)])
	
	c2relCounts[im] = c2byc3
	c3relCounts[im] = c3byc2

# Compile together into one big dict for each image that has
# image number as the key, with an array containing:
# [imName, c1count, c2count-c2rel, c3count-c3rel, max(c2rel, c3rel), 0 if c1ob - sum(all others) is < 0 else c1ob - sum(all others)]
counts = {im : [imNumbers[im], obCounts[im][0], obCounts[im][1] - c2relCounts[im], obCounts[im][2] - c3relCounts[im], max([c2relCounts[im], c3relCounts[im]]), 0 if obCounts[im][0] - sum([obCounts[im][1] - c2relCounts[im], obCounts[im][2] - c3relCounts[im], max([c2relCounts[im], c3relCounts[im]])]) < 0 else obCounts[im][0] - sum([obCounts[im][1] - c2relCounts[im], obCounts[im][2] - c3relCounts[im], max([c2relCounts[im], c3relCounts[im]])])] for im in obCounts.keys()}

# Now generate the percs.
# [imName, c2%, c3%, c2+c3%, unlabeled%]
percs = {im : [imNumbers[im], 0 if counts[im][1] == 0 else counts[im][2] / float(counts[im][1]) * 100, 0 if counts[im][1] == 0 else counts[im][3] / float(counts[im][1]) * 100, 0 if counts[im][1] == 0 else counts[im][4] / float(counts[im][1]) * 100, 0 if counts[im][1] == 0 else counts[im][5] / float(counts[im][1]) * 100] for im in counts.keys()}

print 'Data processed.'

# Now we need to aggregate the percentages for each stain and day.
# 0 green, 1 red, 2 both, 3 neither
for dir in dirs:
	print dir
	# Generate the mean percents for each staining bin and day with std devs.
	dSet = pd.DataFrame.from_dict({x : percs[x] for x in percs.keys() if dir in percs[x][0]}, orient = 'index')
	d0means = list(dSet.ix[dSet[:][0].str.contains('d0')][:][1:].mean())
	d0stds = list(dSet.ix[dSet[:][0].str.contains('d0')][:][1:].std())
	d2means = list(dSet.ix[dSet[:][0].str.contains('d2')][:][1:].mean())
	d2stds = list(dSet.ix[dSet[:][0].str.contains('d2')][:][1:].std())
	d5means = list(dSet.ix[dSet[:][0].str.contains('d5')][:][1:].mean())
	d5stds = list(dSet.ix[dSet[:][0].str.contains('d5')][:][1:].std())
	
	plotMarkerPercs(dir, ['d0', 'd2', 'd5'], 'Percent of Total Nuclei', ['Green', 'Red', 'Both', 'Neither'], d0means, d0stds, d2means, d2stds, d5means, d5stds)
	


