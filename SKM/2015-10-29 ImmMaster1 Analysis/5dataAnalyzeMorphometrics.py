import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import glob
import re
import os
from matplotlib.cm import Blues
from matplotlib.colors import LogNorm

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
titleRE = re.compile('images\\\\((.*-.*)-(.*))')
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
	
def plotPvalueHeatMap():
	return
	
	
print 'Declarations done.'

############################
### Nuclear stain counts ###
############################
# Need to get the image numbers to match with the directories.
imNumbers = {x[0] : str(x[1]) + '\\' + str(x[2]) for x in getData(['ImageNumber', 'Image_PathName_c1', 'Image_FileName_c1'], 'Per_Image')}

# Fetch object counts.
obCounts = getData(['ImageNumber', 'Image_Count_c1ob', 'Image_Count_c2ob', 'Image_Count_c3ob'], 'Per_Image')
obCounts = {x[0] : [x[1], x[2], x[3]] for x in obCounts if x[1] > 100}
c2rels = getData(['ImageNumber', 'c2ob_Number_Object_Number', 'c2ob_Children_c3ob_Count', 'c2ob_Parent_c3ob'], 'Per_c2ob')
c3rels = getData(['ImageNumber', 'c3ob_Number_Object_Number', 'c3ob_Children_c2ob_Count', 'c3ob_Parent_c2ob'], 'Per_c3ob')

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



###########################
###### Morphometrics ######
###########################
# List of morphometric features we'll be examining.
featureNames = ['_AreaShape_Area', '_AreaShape_Compactness', '_AreaShape_Eccentricity', '_AreaShape_Extent', '_AreaShape_FormFactor',
	'_Neighbors_NumberOfNeighbors_Adjacent', '_Neighbors_PercentTouching_Adjacent',
	'_Location_Center_X', '_Location_Center_Y']
	
# Retrieve on an image level. IgG and PBS excluded here.
#! Need to flip flop c2 and c3 features for MsNes stainings, which is done in imageMetrics2
imageMetrics1 = getData(['ImageNumber'] + ['Mean_c1ob' + x for x in featureNames] + ['StDev_c1ob' + x for x in featureNames] + ['Mean_c2ob' + x for x in featureNames] + ['StDev_c2ob' + x for x in featureNames] + ['Mean_c3ob' + x for x in featureNames] + ['StDev_c3ob' + x for x in featureNames], 'Per_Image')
imageMetrics2 = getData(['ImageNumber'] + ['Mean_c1ob' + x for x in featureNames] + ['StDev_c1ob' + x for x in featureNames] + ['Mean_c3ob' + x for x in featureNames] + ['StDev_c3ob' + x for x in featureNames] + ['Mean_c2ob' + x for x in featureNames] + ['StDev_c2ob' + x for x in featureNames], 'Per_Image')
imageMetrics1 = {imNumbers[x[0]] : x[1:] for x in imageMetrics1 if (x[0] in counts.keys() and not ('PBS' in imNumbers[x[0]]) and not ('IgG' in imNumbers[x[0]]) and not ('MsNes' in imNumbers[x[0]]))}
imageMetrics2 = {imNumbers[x[0]] : x[1:] for x in imageMetrics2 if (x[0] in counts.keys() and not ('PBS' in imNumbers[x[0]]) and not ('IgG' in imNumbers[x[0]]) and ('MsNes' in imNumbers[x[0]]))}
imageMetrics = dict(imageMetrics1.items() + imageMetrics2.items())

defineRE = re.compile('.*images\\\\(.*)-(.*)-(.*)\\\\raw\\\\(d\d)-.*-(c.)')
featureList = ['Mean_c1ob' + x for x in featureNames] + ['StDev_c1ob' + x for x in featureNames] + ['Mean_c2ob' + x for x in featureNames] + ['StDev_c2ob' + x for x in featureNames] + ['Mean_c3ob' + x for x in featureNames] + ['StDev_c3ob' + x for x in featureNames]
condiList = list(set([re.search(titleRE, condi).group(2) for condi in dirs]))
#nucleiIms = ['PBS', 'MsNes', 'IgG', 'RbGATA4', 'RbPax6', 'RbBra']
#oct4Ims = ['MsNes', 'RbGATA4', 'RbPax6', 'RbBra']
#pax6Ims = ['RbPax6']
#brachIms = ['RbBra']
#gata4Ims = ['RbGATA4']


# Compare nuclei properties across conditions, days, dims.
#nucMetrics = pd.DataFrame.from_dict({x : imageMetrics[x] for x in imageMetrics.keys() if re.search(defineRE, x).group(3) in nucleiIms}, orient = 'index')
imMetrics = pd.DataFrame.from_dict(imageMetrics, orient = 'index')
imMetrics.columns = featureList
labels = []
for dim in dims:
	for media in mediaKey.values():
		for day in times:
			labels.append(dim + '-' + media + '-' + day)

comps = {}			
subLabels = ['-c1c1', '-c2c2', '-c2c3', '-c3c2', '-c3c3']
labelRE = re.compile('(\dD)-(.*)-(d\d)')
# For each of the dim-media-day labels.
for label1 in labels:
	# Extract parameters of label.
	dim1 = re.search(labelRE, label1).group(1)
	media1 = re.search(labelRE, label1).group(2)
	day1 = re.search(labelRE, label1).group(3)
	# Make the table container that will be added to the comps dict for later plotting.
	table = {}
	# Compare it to each of the other dim-media-day labels (including itself).
	for label2 in labels:
		# Extract parameters of compared label.
		dim2 = re.search(labelRE, label2).group(1)
		media2 = re.search(labelRE, label2).group(2)
		day2 = re.search(labelRE, label2).group(3)
		# Grab the data subsets that correspond to label1 and label2.
		subset1 = imMetrics[imMetrics.index.to_series().str.contains(dim1 + '-' + media1) & imMetrics.index.to_series().str.contains(day1)]
		subset2 = imMetrics[imMetrics.index.to_series().str.contains(dim2 + '-' + media2) & imMetrics.index.to_series().str.contains(day2)]
		# For each object comparison pair specified in subLabels.
		for subLabel in subLabels:
			key = label2 + subLabel
			row = []
			# Process depending on the object comparison being made.
			# Comparison of nuclei
			if (subLabel == '-c1c1'):
				# Slim down to the relevant object columns
				vals1 = subset1[[col for col in subset1.columns if subLabel[1:3] in col]]
				vals2 = subset2[[col for col in subset2.columns if subLabel[3:5] in col]]
				for feature1, feature2 in zip([f1 for f1 in featureList if subLabel[1:3] in f1], [f2 for f2 in featureList if subLabel[3:5] in f2]):
					# Retrieve the appropriate columns of values.
					fVals1 = np.array([v for v in vals1[feature1].values if (not v == 0 and not np.isnan(v))])
					fVals2 = np.array([v for v in vals2[feature2].values if (not v == 0 and not np.isnan(v))])
					
					# If either of the columns ends up having nothing after removing 0's and NaN's, set p value to 1
					if (np.isnan(ttest_ind(fVals1, fVals2, equal_var = False)[1])):
						row.append(1.0)
					# Else calculate the p value.
					else:
						row.append(ttest_ind(fVals1, fVals2, equal_var = False)[1])
			
			# Comparison of green objects.
			elif (subLabel == '-c2c2'):
				# Slim down to the relevant object columns
				vals1 = subset1[[col for col in subset1.columns if subLabel[1:3] in col]]
				vals2 = subset2[[col for col in subset2.columns if subLabel[3:5] in col]]
				for feature1, feature2 in zip([f1 for f1 in featureList if subLabel[1:3] in f1], [f2 for f2 in featureList if subLabel[3:5] in f2]):
					# Retrieve the appropriate columns of values.
					fVals1 = np.array([v for v in vals1[feature1].values if (not v == 0 and not np.isnan(v))])
					fVals2 = np.array([v for v in vals2[feature2].values if (not v == 0 and not np.isnan(v))])
					
					# If either of the columns ends up having nothing after removing 0's and NaN's, set p value to 1
					if (np.isnan(ttest_ind(fVals1, fVals2, equal_var = False)[1])):
						row.append(1.0)
					# Else calculate the p value.
					else:
						row.append(ttest_ind(fVals1, fVals2, equal_var = False)[1])
			
			# Comparison of green and red objects.
			elif (subLabel == '-c2c3'):
				# Slim down to the relevant object columns
				vals1 = subset1[[col for col in subset1.columns if subLabel[1:3] in col]]
				vals2 = subset2[[col for col in subset2.columns if subLabel[3:5] in col]]
				for feature1, feature2 in zip([f1 for f1 in featureList if subLabel[1:3] in f1], [f2 for f2 in featureList if subLabel[3:5] in f2]):
					# Retrieve the appropriate columns of values.
					fVals1 = np.array([v for v in vals1[feature1].values if (not v == 0 and not np.isnan(v))])
					fVals2 = np.array([v for v in vals2[feature2].values if (not v == 0 and not np.isnan(v))])
					
					# If either of the columns ends up having nothing after removing 0's and NaN's, set p value to 1
					if (np.isnan(ttest_ind(fVals1, fVals2, equal_var = False)[1])):
						row.append(1.0)
					# Else calculate the p value.
					else:
						row.append(ttest_ind(fVals1, fVals2, equal_var = False)[1])
			
			# Comparison of red and green objects.
			elif (subLabel == '-c3c2'):
				# Slim down to the relevant object columns
				vals1 = subset1[[col for col in subset1.columns if subLabel[1:3] in col]]
				vals2 = subset2[[col for col in subset2.columns if subLabel[3:5] in col]]
				for feature1, feature2 in zip([f1 for f1 in featureList if subLabel[1:3] in f1], [f2 for f2 in featureList if subLabel[3:5] in f2]):
					# Retrieve the appropriate columns of values.
					fVals1 = np.array([v for v in vals1[feature1].values if (not v == 0 and not np.isnan(v))])
					fVals2 = np.array([v for v in vals2[feature2].values if (not v == 0 and not np.isnan(v))])
					
					# If either of the columns ends up having nothing after removing 0's and NaN's, set p value to 1
					if (np.isnan(ttest_ind(fVals1, fVals2, equal_var = False)[1])):
						row.append(1.0)
					# Else calculate the p value.
					else:
						row.append(ttest_ind(fVals1, fVals2, equal_var = False)[1])
			
			# Comparison of red and red objects.
			elif (subLabel == '-c3c3'):
				# Slim down to the relevant object columns
				vals1 = subset1[[col for col in subset1.columns if subLabel[1:3] in col]]
				vals2 = subset2[[col for col in subset2.columns if subLabel[3:5] in col]]
				for feature1, feature2 in zip([f1 for f1 in featureList if subLabel[1:3] in f1], [f2 for f2 in featureList if subLabel[3:5] in f2]):
					# Retrieve the appropriate columns of values.
					fVals1 = np.array([v for v in vals1[feature1].values if (not v == 0 and not np.isnan(v))])
					fVals2 = np.array([v for v in vals2[feature2].values if (not v == 0 and not np.isnan(v))])
					
					# If either of the columns ends up having nothing after removing 0's and NaN's, set p value to 1
					if (np.isnan(ttest_ind(fVals1, fVals2, equal_var = False)[1])):
						row.append(1.0)
					# Else calculate the p value.
					else:
						row.append(ttest_ind(fVals1, fVals2, equal_var = False)[1])
						
			table[key] = row
			
			#for idx, feature in enumerate(featureList):
			#	# Need to check the comparison is valid i.e. that the Mean isn't None or the StDev isn't 0.
			#	# Use test = imMetrics[imMetrics.index.to_series().str.contains(dim1 + '-' + media1) & imMetrics.index.to_series().str.contains(day1)]
			#	if (not nucMetrics.ix[[x for x in nucMetrics.index if (dim1 in x and media1 in x and day1 in x)]][idx].isnull().values.any() and not nucMetrics.ix[[y for y in nucMetrics.index if (dim2 in y and media2 in y and day2 in y)]][idx].isnull().values.any()):
			#		row.append(ttest_ind(nucMetrics.ix[[x for x in nucMetrics.index if (dim1 in x and media1 in x and day1 in x)]][idx].values, nucMetrics.ix[[y for y in nucMetrics.index if (dim2 in y and media2 in y and day2 in y)]][idx].values, equal_var=False)[1])
			#	else:
			#		row.append(1)
			
	comps[label1] = table

featureLabels = ['Mean' + x for x in featureNames] + ['Std' + x for x in featureNames]
for title in comps.keys():
	rowNames = sorted(comps[title].keys())
#	sums = [[x, sum(comps[title][x])] for x in rowNames] # For summed p Value sort, highest to low.
#	sums = list(reversed(sorted(sums, key = lambda position: position[1]))) # For summed p Value sort, highest to low.
	sums = [[x, sum(1/np.array(comps[title][x]))] for x in rowNames] # For summed inverse p value sort.
	sums = sorted(sums, key = lambda position: position[1]) # For summed inverse p value sort.
	rowNames = [x[0] for x in sums] # For either sums or inverse sums sorting.

	rowVals = np.array([np.array(comps[title][x]) for x in rowNames])

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(rowVals, norm = LogNorm(rowVals.min(), 1), cmap=Blues, edgecolors = 'k', linewidths=1, alpha=0.8)

	# Format
	fig = plt.gcf()
	fig.set_size_inches(7, 30)

	# turn off the frame
	ax.set_frame_on(False)

	# put the major ticks at the middle of each cell
	ax.set_yticks(np.arange(rowVals.shape[0]) + 0.5, minor=False)
	ax.set_xticks(np.arange(rowVals.shape[1]) + 0.5, minor=False)

	# want a more natural, table-like display
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	ax.set_aspect('auto')

	# note I could have used nba_sort.columns but made "labels" instead
	ax.set_xticklabels(featureLabels, rotation = (45), fontsize = 7, weight = 'bold', va='bottom', ha='left', minor=False)
	ax.set_yticklabels(rowNames, fontsize = 7, weight='bold',  minor=False)

	ax.grid(False)

	# Turn off all the ticks
	ax = plt.gca()

	cbar = plt.colorbar(heatmap)

	for t in ax.xaxis.get_major_ticks():
		t.tick1On = False
		t.tick2On = False
	for t in ax.yaxis.get_major_ticks():
		t.tick1On = False
		t.tick2On = False

	plt.text(-5, -9, title, fontsize = 20, weight = 'bold')
	plt.tight_layout()

	plt.savefig('heatmaps\\' + title + '.png')
	plt.close()

	
# pcolor references
# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
# http://matplotlib.org/1.5.0/examples/pylab_examples/pcolor_log.html

