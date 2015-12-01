# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:33:06 2015

@author: Brian
"""

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import re
    from scipy import stats
    import statsmodels.api as sm
    import os
except ImportError:
    raise ImportError

# load dataframes from csv files
const2d_d1 = pd.read_csv('20150509-Alyssa2D-D1-LD_Plate_4710.csv', index_col=0)
const2d_d3 = pd.read_csv('20150511-AlyssaC2D-D3-LD_Plate_4714.csv', index_col=0)
add2d_d3 = pd.read_csv('20150511-AlyssaA2D-D3-LD_Plate_4718.csv', index_col=0)
with2d_d3 = pd.read_csv('20150511-AlyssaW2D-D3-LD_Plate_4716.csv', index_col=0)
#%%
df_list = [const2d_d1, const2d_d3, add2d_d3, with2d_d3]

for dfi in df_list:
    rowiter = range(len(dfi['path']))
    sitelist = []
    for j in rowiter:
        dfi['path'].iloc[j] = os.path.split(os.path.split(dfi['path'].iloc[j])[0])[1]
        sitename = re.search('A\d{2}_s\d{1,3}', dfi['file'].iloc[j]).group()
        sitelist.append(sitename)
        print j, sitename
    # finish by compiling sitelist into a new dataframe column "site"
    dfi['site'] = sitelist
    dfi.set_index(['site'], drop=False, inplace=True)


#%%
sitemap = np.genfromtxt('SiteMap.csv', delimiter=',', dtype=np.str)
constmap = np.genfromtxt('ConstantMap.csv', delimiter=',', dtype=np.str)
addmap = np.genfromtxt('AddMap.csv', delimiter=',', dtype=np.str)
withmap = np.genfromtxt('WithdrawMap.csv', delimiter=',', dtype=np.str)

#%%
def mapcond_to_site(condarray, sitearray):
    clist = []
    slist = []
    csiter = np.nditer([condarray, sitearray])
    for (c, s) in csiter:
        clist.append(c.item())
        slist.append(s.item())
    cseries = pd.Series(data=clist, index=slist, name='condition')
    return cseries

constseries = mapcond_to_site(constmap, sitemap)
addseries = mapcond_to_site(addmap, sitemap)
withseries = mapcond_to_site(withmap, sitemap)

const2d_d1 = const2d_d1.join(constseries, on='site')
const2d_d3 = const2d_d3.join(constseries, on='site')
add2d_d3 = add2d_d3.join(addseries, on='site')
with2d_d3 = with2d_d3.join(withseries, on='site')

#%%
def split_plus(df):

    splitlist = []
    for idx, row in df.iterrows():

        subconditions = row['condition']
        sublist = [row['site'], subconditions]
        print sublist
        plusmatch = re.search('\+', subconditions)

        if plusmatch:
            print plusmatch.group()
            split = re.split(' *\+ *', subconditions)
            [sublist.append(sp) for sp in split]
        splitlist.append(sublist)

    return splitlist

const2d_d1_split = split_plus(const2d_d1)
const2d_d3_split = split_plus(const2d_d3)
add2d_d3_split = split_plus(add2d_d3)
with2d_d3_split = split_plus(with2d_d3)
#%%
def split_to_df(split, sourcedf):
#    split = const2d_d3_split
    
    splitdf = pd.DataFrame(data=split).set_index(0, drop=True)
    
    colnames = ['150ng/ml_wnt3a', '3um_chir', '0.35um_ijnk7', '0.015um_ijnk7', '0.5um_fcsp', '0.33um_fcsp', '7.5um_fh535', '5um_fh535', '0.25um_sp', 'blank']
    resultsdf = pd.DataFrame(None, columns=colnames)
    
    splititer = splitdf.iterrows()
    for idx, i in splititer:
    
        rowresults = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        itemiter = i.iteritems()
        for idx2, j in itemiter:
    
            if j is None:
                pass
            else:
                jnew = re.sub('[ /-]', '', j.lower())
                if re.search('\+', jnew):
                    pass
                elif re.search('wnt(3a)*', jnew):
                    rowresults[0] = 1
                elif re.search('chir', jnew):
                    rowresults[1] = 1
                elif re.search('0.35um(i)*jnk(7)*', jnew):
                    rowresults[2] = 1
                elif re.search('0.015um(i)*jnk(7)*', jnew):
                    rowresults[3] = 1
                elif re.search('0.5umf[cs]{2}p', jnew):
                    rowresults[4] = 1
                elif re.search('0.33umf[cs]{2}p', jnew):
                    rowresults[5] = 1
                elif re.search('7.5umfh535', jnew):
                    rowresults[6] = 1
                elif re.search('^5umfh535', jnew):
                    rowresults[7] = 1
                elif re.search('0.25umsp', jnew):
                    rowresults[8] = 1
                elif re.search('blank', jnew):
                    rowresults[9] = 1
    
                print idx2, jnew
            print rowresults
            resultsdf.loc[idx] = rowresults        
        
    sourcedf = sourcedf.join(resultsdf)
    return sourcedf

const2d_d1 = split_to_df(const2d_d1_split, const2d_d1)
const2d_d3 = split_to_df(const2d_d3_split, const2d_d3)
add2d_d3 = split_to_df(add2d_d3_split, add2d_d3)
with2d_d3 = split_to_df(with2d_d3_split, with2d_d3)

#%%
# Normalize live, dead, and all by raw_dapi
def norm_livedead(df):
    df['live_norm'] = df['live'].div(df['raw_dapi'], axis='index')
    df['dead_norm'] = df['dead'].div(df['raw_dapi'], axis='index')
    df['all_norm'] = df['all'].div(df['raw_dapi'], axis='index')
    return df

constantd1 = norm_livedead(constantd1)
constant = norm_livedead(constant)
add = norm_livedead(add)
withdraw = norm_livedead(withdraw)


#%%
# Write dataframes to file
#const2d_d1.to_csv('constantd1.csv')
#const2d_d3.to_csv('constant.csv')
#add2d_d3.to_csv('add.csv')
#with2d_d3.to_csv('withdraw.csv')


print "Complete!"

#%%
