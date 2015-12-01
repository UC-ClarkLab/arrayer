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
constantd1 = pd.read_csv('constantd1.csv')
constant = pd.read_csv('constant.csv')
add = pd.read_csv('add.csv')
withdraw = pd.read_csv('withdraw.csv')

#%%

# Set up lists for slicing relevant sections of dataframes
ldcounts = ['raw_dapi', 'live', 'dead', 'all']


#%%
'''
# Set up some descriptive plots
sns.set(style='white', context='poster')
# Histograms of raw counts for all chips
# Constant Day 3
fig1, axs1 = plt.subplots(2, 2)
fig1.suptitle('Constant Day 3', fontsize=38)
axs1[0,0].hist(constant[ldcounts[0]], bins=50, color='blue')
axs1[0,0].set_xlabel('Total Nuclei Counted')
axs1[0,0].set_ylabel('Number of Pillars')

axs1[0,1].hist(constant[ldcounts[1]], bins=50, color='green')
axs1[0,1].set_xlabel('Total Live Objects Counted')
axs1[0,1].set_ylabel('Number of Pillars')

axs1[1,0].hist(constant[ldcounts[2]], bins=50, color='red')
axs1[1,0].set_xlabel('Total Dead Objects Counted')
axs1[1,0].set_ylabel('Number of Pillars')

axs1[1,1].hist(constant[ldcounts[3]], bins=50, color='black')
axs1[1,1].set_xlabel('Total Objects Counted with All Stains')
axs1[1,1].set_ylabel('Number of Pillars')

fig1.tight_layout()
fig1.subplots_adjust(top=0.90)

# Constant Day 1
fig2, axs2 = plt.subplots(2, 2)
fig2.suptitle('Constant Day 1', fontsize=38)
axs2[0,0].hist(constantd1[ldcounts[0]], bins=50, color='blue')
axs2[0,0].set_xlabel('Total Nuclei Counted')
axs2[0,0].set_ylabel('Number of Pillars')

axs2[0,1].hist(constantd1[ldcounts[1]], bins=50, color='green')
axs2[0,1].set_xlabel('Total Live Objects Counted')
axs2[0,1].set_ylabel('Number of Pillars')

axs2[1,0].hist(constantd1[ldcounts[2]], bins=50, color='red')
axs2[1,0].set_xlabel('Total Dead Objects Counted')
axs2[1,0].set_ylabel('Number of Pillars')

axs2[1,1].hist(constantd1[ldcounts[3]], bins=50, color='black')
axs2[1,1].set_xlabel('Total Objects Counted with All Stains')
axs2[1,1].set_ylabel('Number of Pillars')

fig2.tight_layout()
fig2.subplots_adjust(top=0.90)


# Add Day 3
fig3, axs3 = plt.subplots(2, 2)
fig3.suptitle('Add Day 3', fontsize=38)
axs3[0,0].hist(add[ldcounts[0]], bins=50, color='blue')
axs3[0,0].set_xlabel('Total Nuclei Counted')
axs3[0,0].set_ylabel('Number of Pillars')

axs3[0,1].hist(add[ldcounts[1]], bins=50, color='green')
axs3[0,1].set_xlabel('Total Live Objects Counted')
axs3[0,1].set_ylabel('Number of Pillars')

axs3[1,0].hist(add[ldcounts[2]], bins=50, color='red')
axs3[1,0].set_xlabel('Total Dead Objects Counted')
axs3[1,0].set_ylabel('Number of Pillars')

axs3[1,1].hist(add[ldcounts[3]], bins=50, color='black')
axs3[1,1].set_xlabel('Total Objects Counted with All Stains')
axs3[1,1].set_ylabel('Number of Pillars')

fig3.tight_layout()
fig3.subplots_adjust(top=0.90)

# Withdraw Day 3
fig4, axs4 = plt.subplots(2, 2)
fig4.suptitle('Withdraw Day 3', fontsize=38)
axs4[0,0].hist(withdraw[ldcounts[0]], bins=50, color='blue')
axs4[0,0].set_xlabel('Total Nuclei Counted')
axs4[0,0].set_ylabel('Number of Pillars')

axs4[0,1].hist(withdraw[ldcounts[1]], bins=50, color='green')
axs4[0,1].set_xlabel('Total Live Objects Counted')
axs4[0,1].set_ylabel('Number of Pillars')

axs4[1,0].hist(withdraw[ldcounts[2]], bins=50, color='red')
axs4[1,0].set_xlabel('Total Dead Objects Counted')
axs4[1,0].set_ylabel('Number of Pillars')

axs4[1,1].hist(withdraw[ldcounts[3]], bins=50, color='black')
axs4[1,1].set_xlabel('Total Objects Counted with All Stains')
axs4[1,1].set_ylabel('Number of Pillars')

fig4.tight_layout()
fig4.subplots_adjust(top=0.90)
'''
#%%
'''
# Summary
fig, axs = plt.subplots(2, 2)
fig.suptitle('2D Chip Summary:', fontsize=38)
axs[0,0].hist(constantd1[ldcounts[0]], bins=50, color='gray', alpha=0.5, label='C_D1')
axs[0,0].hist(constant[ldcounts[0]], bins=50, color='blue', alpha=0.5, label='C_D3')
axs[0,0].hist(add[ldcounts[0]], bins=50, color='green', alpha=0.5, label='A_D3')
axs[0,0].hist(withdraw[ldcounts[0]], bins=50, color='yellow', alpha=0.5, label='W_D3')
axs[0,0].set_xlabel('Total Nuclei Counted')
axs[0,0].set_ylabel('Number of Pillars')

axs[0,1].hist(constantd1[ldcounts[1]], bins=50, color='gray', alpha=0.5, label='C_D1')
axs[0,1].hist(constant[ldcounts[1]], bins=50, color='blue', alpha=0.5, label='C_D3')
axs[0,1].hist(add[ldcounts[1]], bins=50, color='green', alpha=0.5, label='A_D3')
axs[0,1].hist(withdraw[ldcounts[1]], bins=50, color='yellow', alpha=0.5, label='W_D3')
axs[0,1].set_xlabel('Total Live Objects Counted')
axs[0,1].set_ylabel('Number of Pillars')

axs[1,0].hist(constantd1[ldcounts[2]], bins=50, color='gray', alpha=0.5, label='C_D1')
axs[1,0].hist(constant[ldcounts[2]], bins=50, color='blue', alpha=0.5, label='C_D3')
axs[1,0].hist(add[ldcounts[2]], bins=50, color='green', alpha=0.5, label='A_D3')
axs[1,0].hist(withdraw[ldcounts[2]], bins=50, color='yellow', alpha=0.5, label='W_D3')
axs[1,0].set_xlabel('Total Dead Objects Counted')
axs[1,0].set_ylabel('Number of Pillars')

axs[1,1].hist(constantd1[ldcounts[3]], bins=50, color='gray', alpha=0.5, label='C_D1')
axs[1,1].hist(constant[ldcounts[3]], bins=50, color='blue', alpha=0.5, label='C_D3')
axs[1,1].hist(add[ldcounts[3]], bins=50, color='green', alpha=0.5, label='A_D3')
axs[1,1].hist(withdraw[ldcounts[3]], bins=50, color='yellow', alpha=0.5, label='W_D3')
axs[1,1].set_xlabel('Total Objects Counted with All Stains')
axs[1,1].set_ylabel('Number of Pillars')

axs[0,1].legend(loc='upper right')
fig.tight_layout()
fig.subplots_adjust(top=0.90)
'''
#%%

# Histograms of normalized counts for all chips
#sns.set(context='poster', style='white', palette='gray')

ldnormlist = ['live_norm', 'dead_norm', 'all_norm']

'''
# Constant Day 1
fig5, axs5 = plt.subplots(1, 4)
fig5.suptitle('Constant Day 1', fontsize=38)
sns.distplot(constantd1['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs5[0], axlabel='Percent Live')
sns.distplot(constantd1['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs5[1], axlabel='Percent Dead')
sns.distplot(constantd1['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs5[2], axlabel='Percent All Stains')

sns.distplot(constantd1['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs5[3])
sns.distplot(constantd1['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs5[3])
sns.distplot(constantd1['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs5[3])
axs5[3].set_xlabel('Overlay')

axs5[0].set_ylabel('Number of Pillars')
fig5.tight_layout()
fig5.subplots_adjust(top=0.90)

# Constant Day 3
fig6, axs6 = plt.subplots(1, 4)
fig6.suptitle('Constant Day 3', fontsize=38)
sns.distplot(constant['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs6[0], axlabel='Percent Live')
sns.distplot(constant['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs6[1], axlabel='Percent Dead')
sns.distplot(constant['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs6[2], axlabel='Percent All Stains')

sns.distplot(constant['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs6[3])
sns.distplot(constant['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs6[3])
sns.distplot(constant['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs6[3])
axs6[3].set_xlabel('Overlay')

axs6[0].set_ylabel('Number of Pillars')
fig6.tight_layout()
fig6.subplots_adjust(top=0.90)

# Add Day 3
fig7, axs7 = plt.subplots(1, 4)
fig7.suptitle('Add Day 3', fontsize=38)
sns.distplot(add['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs7[0], axlabel='Percent Live')
sns.distplot(add['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs7[1], axlabel='Percent Dead')
sns.distplot(add['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs7[2], axlabel='Percent All Stains')

sns.distplot(add['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs7[3])
sns.distplot(add['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs7[3])
sns.distplot(add['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs7[3])
axs7[3].set_xlabel('Overlay')

axs7[0].set_ylabel('Number of Pillars')
fig7.tight_layout()
fig7.subplots_adjust(top=0.90)

# Withdraw Day 3
fig8, axs8 = plt.subplots(1, 4)
fig8.suptitle('Withdraw Day 3', fontsize=38)
sns.distplot(withdraw['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs8[0], axlabel='Percent Live')
sns.distplot(withdraw['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs8[1], axlabel='Percent Dead')
sns.distplot(withdraw['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs8[2], axlabel='Percent All Stains')

sns.distplot(withdraw['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs8[3])
sns.distplot(withdraw['dead_norm'].dropna(), kde=False, bins=10, color='red', ax=axs8[3])
sns.distplot(withdraw['all_norm'].dropna(), kde=False, bins=10, color='black', ax=axs8[3])
axs8[3].set_xlabel('Overlay')

axs8[0].set_ylabel('Number of Pillars')
fig8.tight_layout()
fig8.subplots_adjust(top=0.90)
'''
#%%
'''
# Summary

#greens = sns.color_palette(sns.dark_palette("green"), 4)
#reds = sns.color_palette(sns.dark_palette("red"), 4)
#blacks = sns.color_palette(sns.dark_palette("black"), 4)

fig9, axs9 = plt.subplots(1, 3)
fig9.suptitle('2D Chip Summary', fontsize=38)
sns.distplot(constantd1['live_norm'].dropna(), kde=False, bins=10, color='gray', ax=axs9[0], label='Constant Day 1')
sns.distplot(constant['live_norm'].dropna(), kde=False, bins=10, color='blue', ax=axs9[0], label='Constant Day 3')
sns.distplot(add['live_norm'].dropna(), kde=False, bins=10, color='green', ax=axs9[0], label='Add Day 3')
sns.distplot(withdraw['live_norm'].dropna(), kde=False, bins=10, color='yellow', ax=axs9[0], label='Withdraw Day 3')
axs9[0].set_xlabel('Percent Live')
axs9[0].set_ylabel('Number of Pillars')

sns.distplot(constantd1['dead_norm'].dropna(), kde=False, bins=10, color='gray', ax=axs9[1], label='Constant Day 1')
sns.distplot(constant['dead_norm'].dropna(), kde=False, bins=10, color='blue', ax=axs9[1], label='Constant Day 3')
sns.distplot(add['dead_norm'].dropna(), kde=False, bins=10, color='green', ax=axs9[1], label='Add Day 3')
sns.distplot(withdraw['dead_norm'].dropna(), kde=False, bins=10, color='yellow', ax=axs9[1], label='Withdraw Day 3')
axs9[1].set_xlabel('Percent Dead')

sns.distplot(constantd1['all_norm'].dropna(), kde=False, bins=10, color='gray', ax=axs9[2], label='Constant Day 1')
sns.distplot(constant['all_norm'].dropna(), kde=False, bins=10, color='blue', ax=axs9[2], label='Constant Day 3')
sns.distplot(add['all_norm'].dropna(), kde=False, bins=10, color='green', ax=axs9[2], label='Add Day 3')
sns.distplot(withdraw['all_norm'].dropna(), kde=False, bins=10, color='yellow', ax=axs9[2], label='Withdraw Day 3')
axs9[2].set_xlabel('Percent All Stains')

#axs9[0].legend(loc='upper right')
axs9[1].legend(loc='upper right')
#axs9[2].legend(loc='upper right')
sns.axes_style({'legend.frameon': True, 'legend.numpoints': 4, 'legend.scatterpoints': 4})
fig9.tight_layout()
fig9.subplots_adjust(top=0.90)
'''
#%%


#cgrouped = constant.groupby('condition')

def describe_groups(df, grouper):
    '''Maps pandas.DataFrame.describe() to group names. Returns a dictionary of {name:stats} pairs.'''
    grouped = df.groupby(grouper)
    gdict = dict()
    for name, group in grouped:
#        print name
        gdict[name] = group.describe()
    return gdict

cstd1stats = describe_groups(constantd1, 'condition')
cststats = describe_groups(constant, 'condition')
addstats = describe_groups(add, 'condition')
withstats = describe_groups(withdraw, 'condition')



#%%
# Boxplots 1: Live stain
'''
# Constant Day 3
fig10, axs10 = plt.subplots(1,1, figsize=(20, 8))
constant.boxplot(column='live_norm', by='condition', rot=90, grid=False, ax=axs10)
axs10.set_ylabel('Percent Live', fontsize=14)
axs10.set_xlabel("")
axs10.set_title('Constant Day 3 Viability', fontsize=20)
sns.despine()
fig10.suptitle("")
fig10.tight_layout()

# Constant Day 1
fig11, axs11 = plt.subplots(1,1, figsize=(20, 8))
constantd1.boxplot(column='live_norm', by='condition', rot=90, grid=False, ax=axs11)
axs11.set_ylabel('Percent Live', fontsize=14)
axs11.set_xlabel("")
axs11.set_title('Constant Day 1 Viability', fontsize=20)
sns.despine()
fig11.suptitle("")
fig11.tight_layout()

# Add Day 3
fig12, axs12 = plt.subplots(1,1, figsize=(20, 8))
add.boxplot(column='live_norm', by='condition', rot=90, grid=False, ax=axs12)
axs12.set_ylabel('Percent Live', fontsize=14)
axs12.set_xlabel("")
axs12.set_title('Add Day 3 Viability', fontsize=20)
sns.despine()
fig12.suptitle("")
fig12.tight_layout()

# Withdraw Day 3
fig13, axs13 = plt.subplots(1,1, figsize=(20, 8))
withdraw.boxplot(column='live_norm', by='condition', rot=90, grid=False, ax=axs13)
axs13.set_ylabel('Percent Live', fontsize=14)
axs13.set_xlabel("")
axs13.set_title('Withdraw Day 3 Viability', fontsize=20)
sns.despine()
fig13.suptitle("")
fig13.tight_layout()
'''
#%%
# Boxplots 2: Dead stain
'''
# Constant Day 3
fig14, axs14 = plt.subplots(1,1, figsize=(20, 8))
constant.boxplot(column='dead_norm', by='condition', rot=90, grid=False, ax=axs14)
axs14.set_ylabel('Percent Dead', fontsize=14)
axs14.set_ylim([0, 1.0])
axs14.set_xlabel("")
axs14.set_title('Constant Day 3 Death', fontsize=20)
sns.despine()
fig14.suptitle("")
fig14.tight_layout()

# Constant Day 1
fig15, axs15 = plt.subplots(1,1, figsize=(20, 8))
constantd1.boxplot(column='dead_norm', by='condition', rot=90, grid=False, ax=axs15)
axs15.set_ylabel('Percent Dead', fontsize=14)
axs15.set_ylim([0, 1.0])
axs15.set_xlabel("")
axs15.set_title('Constant Day 1 Death', fontsize=20)
sns.despine()
fig15.suptitle("")
fig15.tight_layout()

# Add Day 3
fig16, axs16 = plt.subplots(1,1, figsize=(20, 8))
add.boxplot(column='dead_norm', by='condition', rot=90, grid=False, ax=axs16)
axs16.set_ylabel('Percent Dead', fontsize=14)
axs16.set_ylim([0, 1.0])
axs16.set_xlabel("")
axs16.set_title('Add Day 3 Death', fontsize=20)
sns.despine()
fig16.suptitle("")
fig16.tight_layout()

# Withdraw Day 3
fig17, axs17 = plt.subplots(1,1, figsize=(20, 8))
withdraw.boxplot(column='dead_norm', by='condition', rot=90, grid=False, ax=axs17)
axs17.set_ylabel('Percent Dead', fontsize=14)
axs17.set_ylim([0, 1.0])
axs17.set_xlabel("")
axs17.set_title('Withdraw Day 3 Death', fontsize=20)
sns.despine()
fig17.suptitle("")
fig17.tight_layout()
'''
#%%
# Boxplots 3: All stains
'''
# Constant Day 3
fig14, axs14 = plt.subplots(1,1, figsize=(20, 8))
constant.boxplot(column='all_norm', by='condition', rot=90, grid=False, ax=axs14)
axs14.set_ylabel('Percent Costained with All Stains', fontsize=14)
axs14.set_ylim([0, 1.0])
axs14.set_xlabel("")
axs14.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()
fig14.suptitle("")
fig14.tight_layout()

# Constant Day 1
fig15, axs15 = plt.subplots(1,1, figsize=(20, 8))
constantd1.boxplot(column='all_norm', by='condition', rot=90, grid=False, ax=axs15)
axs15.set_ylabel('Percent Costained with All Stains', fontsize=14)
axs15.set_ylim([0, 1.0])
axs15.set_xlabel("")
axs15.set_title('Constant Day 1 All Stains', fontsize=20)
sns.despine()
fig15.suptitle("")
fig15.tight_layout()

# Add Day 3
fig16, axs16 = plt.subplots(1,1, figsize=(20, 8))
add.boxplot(column='all_norm', by='condition', rot=90, grid=False, ax=axs16)
axs16.set_ylabel('Percent Costained with All Stains', fontsize=14)
axs16.set_ylim([0, 1.0])
axs16.set_xlabel("")
axs16.set_title('Add Day 3 All Stains', fontsize=20)
sns.despine()
fig16.suptitle("")
fig16.tight_layout()

# Withdraw Day 3
fig17, axs17 = plt.subplots(1,1, figsize=(20, 8))
withdraw.boxplot(column='all_norm', by='condition', rot=90, grid=False, ax=axs17)
axs17.set_ylabel('Percent Costained with All Stains', fontsize=14)
axs17.set_ylim([0, 1.0])
axs17.set_xlabel("")
axs17.set_title('Withdraw Day 3 All Stains', fontsize=20)
sns.despine()
fig17.suptitle("")
fig17.tight_layout()
'''


#%%
'''
#constant.groupby('3um_chir').get_group(1.0)

xlab_ax18 = 'Presence of 3uM CHIR'

fig18, axs18 = plt.subplots(3,4, figsize=(16, 12))

constantd1.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='3um_chir', rot=90, grid=False, ax=axs18[[0, 1, 2], 0])
constant.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='3um_chir', rot=90, grid=False, ax=axs18[[0, 1, 2], 1])
add.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='3um_chir', rot=90, grid=False, ax=axs18[[0, 1, 2], 2])
withdraw.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='3um_chir', rot=90, grid=False, ax=axs18[[0, 1, 2], 3])
axs18[0, 0].set_ylabel('Percent Live', fontsize=14)
axs18[1, 0].set_ylabel('Percent Dead', fontsize=14)
axs18[2, 0].set_ylabel('Percent All', fontsize=14)

axs18[0, 0].set_ylim([0, 1.0])
axs18[0, 1].set_ylim([0, 1.0])
axs18[0, 2].set_ylim([0, 1.0])
axs18[0, 3].set_ylim([0, 1.0])

axs18[0, 0].set_title('Constant Day 1', fontsize=18)
axs18[0, 1].set_title('Constant Day 3', fontsize=18)
axs18[0, 2].set_title('Add Day 3', fontsize=18)
axs18[0, 3].set_title('Withdraw Day 3', fontsize=18)

axs18[0, 0].set_xlabel(xlab_ax18)
axs18[0, 1].set_xlabel(xlab_ax18)
axs18[0, 2].set_xlabel(xlab_ax18)
axs18[0, 3].set_xlabel(xlab_ax18)

axs18[1, 0].set_ylim([0, 1.0])
axs18[1, 1].set_ylim([0, 1.0])
axs18[1, 2].set_ylim([0, 1.0])
axs18[1, 3].set_ylim([0, 1.0])

axs18[1, 0].set_title('')
axs18[1, 1].set_title('')
axs18[1, 2].set_title('')
axs18[1, 3].set_title('')

axs18[1, 0].set_xlabel(xlab_ax18)
axs18[1, 1].set_xlabel(xlab_ax18)
axs18[1, 2].set_xlabel(xlab_ax18)
axs18[1, 3].set_xlabel(xlab_ax18)

axs18[2, 0].set_ylim([0, 1.0])
axs18[2, 1].set_ylim([0, 1.0])
axs18[2, 2].set_ylim([0, 1.0])
axs18[2, 3].set_ylim([0, 1.0])

axs18[2, 0].set_title('')
axs18[2, 1].set_title('')
axs18[2, 2].set_title('')
axs18[2, 3].set_title('')

axs18[2, 0].set_xlabel(xlab_ax18)
axs18[2, 1].set_xlabel(xlab_ax18)
axs18[2, 2].set_xlabel(xlab_ax18)
axs18[2, 3].set_xlabel(xlab_ax18)

#a.set_ylim([0, 1.0])
#a.set_xlabel("")
#a.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()
fig18.suptitle("3 uM CHIR Effects on Viability", fontsize=20)
fig18.tight_layout()
fig18.subplots_adjust(top=0.92)
'''
#%%
'''
#constant.groupby('150ng/ml_wnt3a').get_group(1.0)

xlab_ax18 = 'Presence of 150 ng/ml Wnt3a'

fig18, axs18 = plt.subplots(3,4, figsize=(16, 12))

constantd1.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='150ng/ml_wnt3a', rot=90, grid=False, ax=axs18[[0, 1, 2], 0])
constant.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='150ng/ml_wnt3a', rot=90, grid=False, ax=axs18[[0, 1, 2], 1])
add.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='150ng/ml_wnt3a', rot=90, grid=False, ax=axs18[[0, 1, 2], 2])
withdraw.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by='150ng/ml_wnt3a', rot=90, grid=False, ax=axs18[[0, 1, 2], 3])
axs18[0, 0].set_ylabel('Percent Live', fontsize=14)
axs18[1, 0].set_ylabel('Percent Dead', fontsize=14)
axs18[2, 0].set_ylabel('Percent All', fontsize=14)

axs18[0, 0].set_ylim([0, 1.0])
axs18[0, 1].set_ylim([0, 1.0])
axs18[0, 2].set_ylim([0, 1.0])
axs18[0, 3].set_ylim([0, 1.0])

axs18[0, 0].set_title('Constant Day 1', fontsize=18)
axs18[0, 1].set_title('Constant Day 3', fontsize=18)
axs18[0, 2].set_title('Add Day 3', fontsize=18)
axs18[0, 3].set_title('Withdraw Day 3', fontsize=18)

axs18[0, 0].set_xlabel(xlab_ax18)
axs18[0, 1].set_xlabel(xlab_ax18)
axs18[0, 2].set_xlabel(xlab_ax18)
axs18[0, 3].set_xlabel(xlab_ax18)

axs18[1, 0].set_ylim([0, 1.0])
axs18[1, 1].set_ylim([0, 1.0])
axs18[1, 2].set_ylim([0, 1.0])
axs18[1, 3].set_ylim([0, 1.0])

axs18[1, 0].set_title('')
axs18[1, 1].set_title('')
axs18[1, 2].set_title('')
axs18[1, 3].set_title('')

axs18[1, 0].set_xlabel(xlab_ax18)
axs18[1, 1].set_xlabel(xlab_ax18)
axs18[1, 2].set_xlabel(xlab_ax18)
axs18[1, 3].set_xlabel(xlab_ax18)

axs18[2, 0].set_ylim([0, 1.0])
axs18[2, 1].set_ylim([0, 1.0])
axs18[2, 2].set_ylim([0, 1.0])
axs18[2, 3].set_ylim([0, 1.0])

axs18[2, 0].set_title('')
axs18[2, 1].set_title('')
axs18[2, 2].set_title('')
axs18[2, 3].set_title('')

axs18[2, 0].set_xlabel(xlab_ax18)
axs18[2, 1].set_xlabel(xlab_ax18)
axs18[2, 2].set_xlabel(xlab_ax18)
axs18[2, 3].set_xlabel(xlab_ax18)

#a.set_ylim([0, 1.0])
#a.set_xlabel("")
#a.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()
fig18.suptitle("150 ng/ml Wnt3a Effects on Viability", fontsize=20)
fig18.tight_layout()
fig18.subplots_adjust(top=0.92)

'''

#%%
'''
xlab_ax18 = 'Concentration of FH535'
ticklabels = ['None', '7.5 uM', '5 uM']

fig18, axs18 = plt.subplots(3,4, figsize=(16, 12))

constantd1.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['7.5um_fh535', '5um_fh535'], rot=90, grid=False, ax=axs18[[0, 1, 2], 0])
constant.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['7.5um_fh535', '5um_fh535'], rot=90, grid=False, ax=axs18[[0, 1, 2], 1])
add.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['7.5um_fh535', '5um_fh535'], rot=90, grid=False, ax=axs18[[0, 1, 2], 2])
withdraw.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['7.5um_fh535', '5um_fh535'], rot=90, grid=False, ax=axs18[[0, 1, 2], 3])
axs18[0, 0].set_ylabel('Percent Live', fontsize=14)
axs18[1, 0].set_ylabel('Percent Dead', fontsize=14)
axs18[2, 0].set_ylabel('Percent All', fontsize=14)

axs18[0, 0].set_ylim([0, 1.0])
axs18[0, 1].set_ylim([0, 1.0])
axs18[0, 2].set_ylim([0, 1.0])
axs18[0, 3].set_ylim([0, 1.0])

axs18[0, 0].set_title('Constant Day 1', fontsize=18)
axs18[0, 1].set_title('Constant Day 3', fontsize=18)
axs18[0, 2].set_title('Add Day 3', fontsize=18)
axs18[0, 3].set_title('Withdraw Day 3', fontsize=18)

axs18[0, 0].set_xlabel(xlab_ax18)
axs18[0, 1].set_xlabel(xlab_ax18)
axs18[0, 2].set_xlabel(xlab_ax18)
axs18[0, 3].set_xlabel(xlab_ax18)

axs18[1, 0].set_ylim([0, 1.0])
axs18[1, 1].set_ylim([0, 1.0])
axs18[1, 2].set_ylim([0, 1.0])
axs18[1, 3].set_ylim([0, 1.0])

axs18[1, 0].set_title('')
axs18[1, 1].set_title('')
axs18[1, 2].set_title('')
axs18[1, 3].set_title('')

axs18[1, 0].set_xlabel(xlab_ax18)
axs18[1, 1].set_xlabel(xlab_ax18)
axs18[1, 2].set_xlabel(xlab_ax18)
axs18[1, 3].set_xlabel(xlab_ax18)

axs18[2, 0].set_ylim([0, 1.0])
axs18[2, 1].set_ylim([0, 1.0])
axs18[2, 2].set_ylim([0, 1.0])
axs18[2, 3].set_ylim([0, 1.0])

axs18[2, 0].set_title('')
axs18[2, 1].set_title('')
axs18[2, 2].set_title('')
axs18[2, 3].set_title('')

axs18[2, 0].set_xlabel(xlab_ax18)
axs18[2, 1].set_xlabel(xlab_ax18)
axs18[2, 2].set_xlabel(xlab_ax18)
axs18[2, 3].set_xlabel(xlab_ax18)

#a.set_ylim([0, 1.0])
#a.set_xlabel("")
#a.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()

axs18[0,0].set_xticklabels(ticklabels, rotation=0)
axs18[0,1].set_xticklabels(ticklabels, rotation=0)
axs18[0,2].set_xticklabels(ticklabels, rotation=0)
axs18[0,3].set_xticklabels(ticklabels, rotation=0)

axs18[1,0].set_xticklabels(ticklabels, rotation=0)
axs18[1,1].set_xticklabels(ticklabels, rotation=0)
axs18[1,2].set_xticklabels(ticklabels, rotation=0)
axs18[1,3].set_xticklabels(ticklabels, rotation=0)

axs18[2,0].set_xticklabels(ticklabels, rotation=0)
axs18[2,1].set_xticklabels(ticklabels, rotation=0)
axs18[2,2].set_xticklabels(ticklabels, rotation=0)
axs18[2,3].set_xticklabels(ticklabels, rotation=0)

fig18.suptitle("FH535 Effects on Viability", fontsize=20)
fig18.tight_layout()
fig18.subplots_adjust(top=0.92)
'''

#%%
'''
xlab_ax18 = 'Concentration of FCSP'
ticklabels = ['None', '0.5 uM', '0.33 uM']

fig18, axs18 = plt.subplots(3,4, figsize=(16, 12))

constantd1.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.5um_fcsp', '0.33um_fcsp'], rot=90, grid=False, ax=axs18[[0, 1, 2], 0])
constant.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.5um_fcsp', '0.33um_fcsp'], rot=90, grid=False, ax=axs18[[0, 1, 2], 1])
add.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.5um_fcsp', '0.33um_fcsp'], rot=90, grid=False, ax=axs18[[0, 1, 2], 2])
withdraw.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.5um_fcsp', '0.33um_fcsp'], rot=90, grid=False, ax=axs18[[0, 1, 2], 3])
axs18[0, 0].set_ylabel('Percent Live', fontsize=14)
axs18[1, 0].set_ylabel('Percent Dead', fontsize=14)
axs18[2, 0].set_ylabel('Percent All', fontsize=14)

axs18[0, 0].set_ylim([0, 1.0])
axs18[0, 1].set_ylim([0, 1.0])
axs18[0, 2].set_ylim([0, 1.0])
axs18[0, 3].set_ylim([0, 1.0])

axs18[0, 0].set_title('Constant Day 1', fontsize=18)
axs18[0, 1].set_title('Constant Day 3', fontsize=18)
axs18[0, 2].set_title('Add Day 3', fontsize=18)
axs18[0, 3].set_title('Withdraw Day 3', fontsize=18)

axs18[0, 0].set_xlabel(xlab_ax18)
axs18[0, 1].set_xlabel(xlab_ax18)
axs18[0, 2].set_xlabel(xlab_ax18)
axs18[0, 3].set_xlabel(xlab_ax18)

axs18[1, 0].set_ylim([0, 1.0])
axs18[1, 1].set_ylim([0, 1.0])
axs18[1, 2].set_ylim([0, 1.0])
axs18[1, 3].set_ylim([0, 1.0])

axs18[1, 0].set_title('')
axs18[1, 1].set_title('')
axs18[1, 2].set_title('')
axs18[1, 3].set_title('')

axs18[1, 0].set_xlabel(xlab_ax18)
axs18[1, 1].set_xlabel(xlab_ax18)
axs18[1, 2].set_xlabel(xlab_ax18)
axs18[1, 3].set_xlabel(xlab_ax18)

axs18[2, 0].set_ylim([0, 1.0])
axs18[2, 1].set_ylim([0, 1.0])
axs18[2, 2].set_ylim([0, 1.0])
axs18[2, 3].set_ylim([0, 1.0])

axs18[2, 0].set_title('')
axs18[2, 1].set_title('')
axs18[2, 2].set_title('')
axs18[2, 3].set_title('')

axs18[2, 0].set_xlabel(xlab_ax18)
axs18[2, 1].set_xlabel(xlab_ax18)
axs18[2, 2].set_xlabel(xlab_ax18)
axs18[2, 3].set_xlabel(xlab_ax18)

#a.set_ylim([0, 1.0])
#a.set_xlabel("")
#a.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()

axs18[0,0].set_xticklabels(ticklabels, rotation=0)
axs18[0,1].set_xticklabels(ticklabels, rotation=0)
axs18[0,2].set_xticklabels(ticklabels, rotation=0)
axs18[0,3].set_xticklabels(ticklabels, rotation=0)

axs18[1,0].set_xticklabels(ticklabels, rotation=0)
axs18[1,1].set_xticklabels(ticklabels, rotation=0)
axs18[1,2].set_xticklabels(ticklabels, rotation=0)
axs18[1,3].set_xticklabels(ticklabels, rotation=0)

axs18[2,0].set_xticklabels(ticklabels, rotation=0)
axs18[2,1].set_xticklabels(ticklabels, rotation=0)
axs18[2,2].set_xticklabels(ticklabels, rotation=0)
axs18[2,3].set_xticklabels(ticklabels, rotation=0)

fig18.suptitle("FCSP Effects on Viability", fontsize=20)
fig18.tight_layout()
fig18.subplots_adjust(top=0.92)
'''

#%%
'''
xlab_ax18 = 'Concentration of iJNK'
ticklabels = ['None', '0.35 uM', '0.015 uM']

fig18, axs18 = plt.subplots(3,4, figsize=(16, 12))

constantd1.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.35um_ijnk7', '0.015um_ijnk7'], rot=90, grid=False, ax=axs18[[0, 1, 2], 0])
constant.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.35um_ijnk7', '0.015um_ijnk7'], rot=90, grid=False, ax=axs18[[0, 1, 2], 1])
add.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.35um_ijnk7', '0.015um_ijnk7'], rot=90, grid=False, ax=axs18[[0, 1, 2], 2])
withdraw.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['0.35um_ijnk7', '0.015um_ijnk7'], rot=90, grid=False, ax=axs18[[0, 1, 2], 3])
axs18[0, 0].set_ylabel('Percent Live', fontsize=14)
axs18[1, 0].set_ylabel('Percent Dead', fontsize=14)
axs18[2, 0].set_ylabel('Percent All', fontsize=14)

axs18[0, 0].set_ylim([0, 1.0])
axs18[0, 1].set_ylim([0, 1.0])
axs18[0, 2].set_ylim([0, 1.0])
axs18[0, 3].set_ylim([0, 1.0])

axs18[0, 0].set_title('Constant Day 1', fontsize=18)
axs18[0, 1].set_title('Constant Day 3', fontsize=18)
axs18[0, 2].set_title('Add Day 3', fontsize=18)
axs18[0, 3].set_title('Withdraw Day 3', fontsize=18)

axs18[0, 0].set_xlabel(xlab_ax18)
axs18[0, 1].set_xlabel(xlab_ax18)
axs18[0, 2].set_xlabel(xlab_ax18)
axs18[0, 3].set_xlabel(xlab_ax18)

axs18[1, 0].set_ylim([0, 1.0])
axs18[1, 1].set_ylim([0, 1.0])
axs18[1, 2].set_ylim([0, 1.0])
axs18[1, 3].set_ylim([0, 1.0])

axs18[1, 0].set_title('')
axs18[1, 1].set_title('')
axs18[1, 2].set_title('')
axs18[1, 3].set_title('')

axs18[1, 0].set_xlabel(xlab_ax18)
axs18[1, 1].set_xlabel(xlab_ax18)
axs18[1, 2].set_xlabel(xlab_ax18)
axs18[1, 3].set_xlabel(xlab_ax18)

axs18[2, 0].set_ylim([0, 1.0])
axs18[2, 1].set_ylim([0, 1.0])
axs18[2, 2].set_ylim([0, 1.0])
axs18[2, 3].set_ylim([0, 1.0])

axs18[2, 0].set_title('')
axs18[2, 1].set_title('')
axs18[2, 2].set_title('')
axs18[2, 3].set_title('')

axs18[2, 0].set_xlabel(xlab_ax18)
axs18[2, 1].set_xlabel(xlab_ax18)
axs18[2, 2].set_xlabel(xlab_ax18)
axs18[2, 3].set_xlabel(xlab_ax18)

#a.set_ylim([0, 1.0])
#a.set_xlabel("")
#a.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()

axs18[0,0].set_xticklabels(ticklabels, rotation=0)
axs18[0,1].set_xticklabels(ticklabels, rotation=0)
axs18[0,2].set_xticklabels(ticklabels, rotation=0)
axs18[0,3].set_xticklabels(ticklabels, rotation=0)

axs18[1,0].set_xticklabels(ticklabels, rotation=0)
axs18[1,1].set_xticklabels(ticklabels, rotation=0)
axs18[1,2].set_xticklabels(ticklabels, rotation=0)
axs18[1,3].set_xticklabels(ticklabels, rotation=0)

axs18[2,0].set_xticklabels(ticklabels, rotation=0)
axs18[2,1].set_xticklabels(ticklabels, rotation=0)
axs18[2,2].set_xticklabels(ticklabels, rotation=0)
axs18[2,3].set_xticklabels(ticklabels, rotation=0)

fig18.suptitle("iJNK Effects on Viability", fontsize=20)
fig18.tight_layout()
fig18.subplots_adjust(top=0.92)
'''

#%%
'''
xlab_ax18 = 'Concentration of Wnt3a/CHIR'
ticklabels = ['None', '150 ng/ml', '3 uM']

fig18, axs18 = plt.subplots(3,4, figsize=(16, 12))

constantd1.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['150ng/ml_wnt3a', '3um_chir'], rot=90, grid=False, ax=axs18[[0, 1, 2], 0])
constant.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['150ng/ml_wnt3a', '3um_chir'], rot=90, grid=False, ax=axs18[[0, 1, 2], 1])
add.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['150ng/ml_wnt3a', '3um_chir'], rot=90, grid=False, ax=axs18[[0, 1, 2], 2])
withdraw.boxplot(column=['live_norm', 'dead_norm', 'all_norm'], by=['150ng/ml_wnt3a', '3um_chir'], rot=90, grid=False, ax=axs18[[0, 1, 2], 3])
axs18[0, 0].set_ylabel('Percent Live', fontsize=14)
axs18[1, 0].set_ylabel('Percent Dead', fontsize=14)
axs18[2, 0].set_ylabel('Percent All', fontsize=14)

axs18[0, 0].set_ylim([0, 1.0])
axs18[0, 1].set_ylim([0, 1.0])
axs18[0, 2].set_ylim([0, 1.0])
axs18[0, 3].set_ylim([0, 1.0])

axs18[0, 0].set_title('Constant Day 1', fontsize=18)
axs18[0, 1].set_title('Constant Day 3', fontsize=18)
axs18[0, 2].set_title('Add Day 3', fontsize=18)
axs18[0, 3].set_title('Withdraw Day 3', fontsize=18)

axs18[0, 0].set_xlabel(xlab_ax18)
axs18[0, 1].set_xlabel(xlab_ax18)
axs18[0, 2].set_xlabel(xlab_ax18)
axs18[0, 3].set_xlabel(xlab_ax18)

axs18[1, 0].set_ylim([0, 1.0])
axs18[1, 1].set_ylim([0, 1.0])
axs18[1, 2].set_ylim([0, 1.0])
axs18[1, 3].set_ylim([0, 1.0])

axs18[1, 0].set_title('')
axs18[1, 1].set_title('')
axs18[1, 2].set_title('')
axs18[1, 3].set_title('')

axs18[1, 0].set_xlabel(xlab_ax18)
axs18[1, 1].set_xlabel(xlab_ax18)
axs18[1, 2].set_xlabel(xlab_ax18)
axs18[1, 3].set_xlabel(xlab_ax18)

axs18[2, 0].set_ylim([0, 1.0])
axs18[2, 1].set_ylim([0, 1.0])
axs18[2, 2].set_ylim([0, 1.0])
axs18[2, 3].set_ylim([0, 1.0])

axs18[2, 0].set_title('')
axs18[2, 1].set_title('')
axs18[2, 2].set_title('')
axs18[2, 3].set_title('')

axs18[2, 0].set_xlabel(xlab_ax18)
axs18[2, 1].set_xlabel(xlab_ax18)
axs18[2, 2].set_xlabel(xlab_ax18)
axs18[2, 3].set_xlabel(xlab_ax18)

#a.set_ylim([0, 1.0])
#a.set_xlabel("")
#a.set_title('Constant Day 3 All Stains', fontsize=20)
sns.despine()

axs18[0,0].set_xticklabels(ticklabels, rotation=0)
axs18[0,1].set_xticklabels(ticklabels, rotation=0)
axs18[0,2].set_xticklabels(ticklabels, rotation=0)
axs18[0,3].set_xticklabels(ticklabels, rotation=0)

axs18[1,0].set_xticklabels(ticklabels, rotation=0)
axs18[1,1].set_xticklabels(ticklabels, rotation=0)
axs18[1,2].set_xticklabels(ticklabels, rotation=0)
axs18[1,3].set_xticklabels(ticklabels, rotation=0)

axs18[2,0].set_xticklabels(ticklabels, rotation=0)
axs18[2,1].set_xticklabels(ticklabels, rotation=0)
axs18[2,2].set_xticklabels(ticklabels, rotation=0)
axs18[2,3].set_xticklabels(ticklabels, rotation=0)

fig18.suptitle("Wnt3a/CHIR Effects on Viability", fontsize=20)
fig18.tight_layout()
fig18.subplots_adjust(top=0.92)
'''
#%%
# End

plt.show()
print "Complete!"
#%%

# Write dataframes to file
#constantd1.to_csv('constantd1.csv')
#constant.to_csv('constant.csv')
#add.to_csv('add.csv')
#withdraw.to_csv('withdraw.csv')

