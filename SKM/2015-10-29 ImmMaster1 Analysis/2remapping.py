import csv
import glob
import re

chipNames = glob.glob('maps\*.csv')

chips = []

for chip in chipNames:
	condiList = ''
	csvFile = open(chip, 'rt')
	for row in csvFile:
		condiList += row
	condiList = condiList[:-1]
	condiList = condiList.replace('\n', ',')
	condiList = condiList.split(',')
	chips.append(condiList)
	
imageList = glob.glob('images\*.tif')
ImmInfoRE = re.compile('images\\\\(\d)Dd(\d)w(\d\d)c(\d)s(\d{1,3}).tif')

# Media key.
mediaKey = {'1':'mTeSR', '2':'APEL', '3':'NeuroDiff', '4':'MesoDiff', '5':'EndoDiff'}

# Stain key.
stainKey = {'1':'MsNes', '2':'RbPax6', '3':'RbBra', '4':'RbGATA4', '5':'IgG', '6':'PBS'}

def condiTransform (name):
	dim, day, well, channel, site = re.search(ImmInfoRE, name).groups()
	chipNum = int(dim)%2 + int(day)/2*2
	col = (int(site)-1)%13
	row = (int(site)-1)/13
	siteNum = (int(well)-1)*182 + col*14 + 13-row
	
	condition = chips[chipNum][siteNum]
	media = mediaKey[condition[0]]
	stain = stainKey[condition[2]]
	
	return dim + 'D-' + media + '-' + stain + '-d' + day + '-c' + channel

condiRE = re.compile('(.*-.*-.*-.*-.*)')
# Make list of conditions.
condiFolders = set([re.search(condiRE, condiTransform(x)).group(1) for x in imageList])
condiCounts = {x : 0 for x in condiFolders}
		
for im in imageList:
	condiCounts[condiTransform(im)] += 1
	newName = 'images\\' + condiTransform(im) + '-' + str(condiCounts[condiTransform(im)])
	os.rename(im, newName + '.tif')
	
imageList = glob.glob('images\*.tif')
folderRE = re.compile('images\\\\(.*)-d\d.*')
folders = set([re.search(folderRE, im).group(1) for im in imageList])

for folder in folders:
	if folder not in glob.glob('images\*'):
		os.mkdir('images\\' + folder)
		
sortRE = re.compile('images\\\\(.*)-(d\d)-(c\d)-(\d{1,2})')

for im in imageList:
	con, day, channel, num = re.search(sortRE, im).groups()
	newName = 'images\\' + con + '\\' + day + '-' + num.zfill(2) + '-' + channel + '.tif'
	os.rename(im, newName)