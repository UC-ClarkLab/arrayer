# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:25:40 2015

@author: Brian
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


spreadsheet = '/20150512_BCP_LiveDead_Measure1_Image.csv'

directory = 'C:/Users/Brian/Desktop/Data For Analysis/2015-05-11 AlyssaScreen1/'

paths = ['20150509-Alyssa2D-D1-LD_Plate_4710']#, '20150511-AlyssaA2D-D3-LD_Plate_4718', '20150511-AlyssaC2D-D3-LD_Plate_4714', '20150511-AlyssaW2D-D3-LD_Plate_4716']

droprows = [1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 131, 144, 156*2 + 12, 156*2 + 13, 156*2 + 25, 156*2 + 26, 156*2 + 38, 156*2 + 39, 156*2 + 51, 156*2 + 52, 156*2 + 64, 156*2 + 65, 156*2 + 77, 156*2 + 78, 156*2 + 90, 156*2 + 91, 156*2 + 103, 156*2 + 104, 156*2 + 116, 156*2 + 117, 156*2 + 129, 156*2 + 130, 156*2 + 142, 156*2 + 143, 156*2 + 155, 156*2 + 156]

sns.set(context="talk", style="white", palette="gray")

for p in paths:
    df = pd.DataFrame.from_csv(directory + p + '/CPResults/' + spreadsheet, index_col="ImageNumber")
    df.drop(df.columns[range(0, 19)], axis=1, inplace=True)
    # drop empty pillars!!
    df.drop(df.loc[droprows].index, axis=0, inplace=True)
    # Plot live cell count vs dead cell count
    sns.jointplot("Count_Live", "Count_Dead", data=df)#, xlim=500, ylim=500)

