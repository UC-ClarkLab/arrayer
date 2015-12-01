# -*- coding: utf-8 -*-
"""
Created on Wed May 06 15:13:37 2015

@author: Brian Perea
"""

'''Imports'''

# Notes:
# the y value needs to DECREASE from 57.0 max to 4.5 min
# the x value needs to INCREASE from 219.0 min to 235.5 max
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import xray
    import seaborn as sns # pandas aware plotting library
    import csv
    import re
    import string

except ImportError:
    raise ImportError("Problem importing required module.")


# %%

'''Class definitions'''


class PlateSet:
    '''Class for combining cell culture plate info and data.'''
    def __init__(self):
        self.plates = dict()


class Plate:
    '''Class for specifying cell culture plate maps.'''
    def __init__(self, treat_csv, notes, shape=((38, 14))):
        self.shape = shape  # plate format
        self.treatments = pd.DataFrame.from_csv(treat_csv)
        # row = treatment;
        # columns = components;
        # each item has tuple (quantity, units, start, stop)
        self.map = xray.DataArray()
        self.metadata = dict(notes)

        # I want to map a pandas dataframe with tuples to
        # an xray.DataArray with depth where each unit depth
        # is one component and each entry is a list of quantities
        # for that component over the length of the assay.


class Well:
    '''
    Class for specifying attributes of a well in a multiwell
    plate.
    '''
    def __init__(self):
        self.location = ()
        self.data = xray.DataArray()
        self.metadata = dict()


class Treatments:
    '''
    Class for specifying details for experimental treatments.
    '''
    def __init__(self, treat_csv):
        self.treatments = pd.DataFrame.from_csv(treat_csv)

        self.metadata = dict()


# %%

def genconditionsdf():
    pass


def enum_conditions():
    pass


def mapconditions(treatments_df, no_chips=1, chipspecs='full', save_maps=True, save_pos=True):
    '''Use to map experimental conditions and replicates to randomized
    chip positions.

    Input:
    ---
    treatments_df = Pandas df where each row is a unique treatment,
        column [1] is the number of replicates, and columns [2:] are
        dosage values for all components of the experiment.

    chipspeces = 'full', specifies to use full (38, 14) plate format
        (532 wells including border wells)
             = 'inner', specifies to use only inner (36, 12) 432 wells
             = (y, x), specifies a custom plate format

    Output:
    ---
    .txt - N text files containing positions for each separate condition
    .png - N images showing the locations for each separate condition on chip
    where N = # of distinct conditions
    '''
    # - Specify chip parameters -
    # Center-to-center distance between pillars/wells is 1.5 mm +/- 5 microns
    increment = 1.5 #mm

    '''
    ONLY "full" IS IMPLEMENTED

    If "full" use xA01, yA01
    elif "inner" use xA02, yA02

    Also adjust x by -0.2 and y by +0.2 for best alignment.
    Verify this works best.
    '''
    # The x-position in mm of pillar/well A01
    xA01 = 217.5 - 0.2  # mm
    xA02 = 219.0 - 0.2  # mm this is the first interior x position
    # all x positions must INCREASE from xA02 to 235.5 MAX

    # The y-position in mm of pillar/well A01
    yA01 = 58.5 + 0.2  # mm
    yA02 = 57.0 + 0.2  # mm this is the first interior y position
    # all y positions must DECREASE from yA02 to 4.5 MIN

    # The x-distance between chips
    xchipsep = 27.65 #mm

    # The y-distance between chips
    ychipsep = -0.01 #mm

    # - Set up print -

    # Available pillars/wells on chip
    if chipspecs == 'full':
        dimy, dimx = (38, 14)
        startx = xA01
        starty = yA01
    elif chipspecs == 'inner':
        dimy, dimx = (36, 12)
        startx = xA02
        starty = yA02
    else:
        dimy, dimx = (chipspecs)
        startx = xA01
        starty = yA01
    try:
         available = np.zeros((dimy, dimx))
    except TypeError:
        raise TypeError("Please specify the chip dimensions as 'full', 'inner', or a custom tuple (y, x) of integer values.")

    # Number of wells
    number = dimy * dimx
    spots = range(0, number, 1)

    # Check available == number of replicates in df
    if number < treatments_df['Reps'].sum():
        print "Please ensure the number of treatments is less than or equal to the number of wells."
        return

    # Number of components = component columns - 1 replicates column
    components = treatments_df.shape[1] - 1

    # Specify which columns you want positions for here!!
    treatments = treatments_df.keys()[1:].astype(str)

    # Row mask
    rowMask = list(xrange(dimx))*dimy

    #Column mask
    columnMask = sorted(list(xrange(dimy))*dimx)

    # The x-position of each column for all rows is the same
    xgen = list((x * increment + startx for x in rowMask))

    # The y-position of each row for all columns is the same
    ygen = list((starty - y * increment for y in columnMask))

   # duplicate rows based on number of replicates until we have 532 rows

    for i, item in enumerate(treatments_df['Reps']):

        for j in range(0, item-1):

            treatments_df = treatments_df.append(treatments_df.iloc[i, :])

    components = treatments_df.shape[1]-1

    treatments_df.sort(axis=0, inplace=True)

    xchip_keys = []
    ychip_keys = []

    for c in range(no_chips):

        print 'Generating positions for chip {0}...'.format(c)
        xchip_name = 'X{0}'.format(str(c))
        ychip_name = 'Y{0}'.format(str(c))

        treatments_df.insert(1, xchip_name, [x + c * xchipsep for x in xgen])
        treatments_df.insert(1, ychip_name, [y + c * ychipsep for y in ygen])

        xchip_keys.append(xchip_name)
        ychip_keys.append(ychip_name)
    print 'Here are the keys!'
    print xchip_keys
    print ychip_keys

#    print 'initial shape:', treatments_df.shape
    # Save dataframe with positions to file
    treatments_df.to_csv('chip_plan_gen.csv')

    array = treatments_df.iloc[:, 1:].values

    print treatments_df.iloc[:, 1:].head()

#    print 'Number of Samples', len(array)
#    print '(Number of Samples, Treatments)', array.shape

    # shuffle here
#    np.random.shuffle(array)
#    print array, array.shape
    array = np.expand_dims(array, axis=0).reshape(dimy, dimx, array.shape[1]) # conditions, row, components

#    print type(array[0,0,3])

    dataarray = xray.DataArray(array, dims=['y', 'x', 'z'], coords={'z':treatments_df.columns[1:].values}, name=str(chipspecs))


#    print treatments
    idx = dataarray.loc[dict(z=xchip_keys)]
    idy = dataarray.loc[dict(z=ychip_keys)]

    # Create dictionary to map letters to numbers
    # for visualization of categorical conditions on chip

    def sum_letters(sum_str):
        '''
        Return the sum of the value of letters in a string.

        e.g.

        string = 'abc'

        value_dict = dict(zip(['abc'], [0, 1, 2]))

        value_dict['a'] + value_dict['b'] + value_dict['c'] == 3

        Returns: 3
        '''
        # build dict to map letters to values based on alphabetical order
        letters = string.ascii_lowercase
        numbers = range(0, len(letters))
        pairs = zip(letters, numbers)
        letter_dict = dict(pairs)

        # convert provided string to lower case
        try:
            sum_str = string.lower(sum_str)
        except:
            return sum_str

        values = []
        for l in sum_str:
            try:
                values.append(letter_dict[l])
            except:
                values.append(int(l))

        return sum(values)

    # vectorize function to efficiently run over an array
    vsum_letters = np.vectorize(sum_letters)

    for k, treat in enumerate(treatments):
        t = dataarray.loc[dict(z = treat)]
        tname = re.sub('\s', '-', treat)

        # Extract boolean locations for each treatment
        # and save those as images
        try:
            tnew = np.array(t, dtype=np.uint32)
            if save_maps:
                plt.imsave('{0}.png'.format(tname), tnew)

        except:
            tnew = vsum_letters(t)
            if save_maps:
                plt.imsave('{0}.png'.format(tname), tnew)

        if save_pos:
            for l, unq in enumerate(np.unique(t)):

                tbool = t == unq

                if save_maps:
                    plt.imsave('{0}{1}{2}.png'.format(tname, '_', str(unq)), tbool)
    #            print unq, treat
                xpos = []
                for xkey in xchip_keys:
                    xbool = treatments_df['Treatment'].values == unq
                    xextract = np.extract(xbool, treatments_df[xkey])
                    xpos.extend(xextract.tolist())

                ypos = []
                for ykey in ychip_keys:
                    ybool = treatments_df['Treatment'].values == unq
                    yextract = np.extract(ybool, treatments_df[ykey])
                    ypos.extend(yextract.tolist())

                xname = '{0}{1}{2}_x.txt'.format(tname, '_', str(unq))
                yname = '{0}{1}{2}_y.txt'.format(tname, '_', str(unq))

                with open(xname, 'w') as xf:
                    for x in xpos:
                        xf.write('{0}, '.format(x))

                with open(yname, 'w') as yf:
                    for y in ypos:
                        yf.write('{0}, '.format(y))

                print xname
                print yname

    return (idx, idy, treatments_df, array, dataarray)

    '''
    # From a list of conditions containing duplicates, find the
    # unique condition names and map them to a dictionary
    unq = np.unique(array) # iterate on this, call from dct dictionary!!
    unq = unq[unq != '*']

    # Generate the corresponding map of volumes to be printed
    dct = dict()
    for i in unq:

        dct[i] = np.sum(array == i, axis=2)

    #    print dct[i]

        xposi = idx[dct[i] != 0]  # only the important x positions
        yposi = idy[dct[i] != 0]  # only the important y positions
        voli = (dct[i] * 0.266)[(dct[i] * 0.266) != 0]  # volumes to be printed
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
    '''
# %%

'''Script'''

if __name__ == "__main__":

    df_base = pd.DataFrame.from_csv('chip_plan.csv', index_col=0).fillna(value=0)

    (idx, idy, df, array, dataarray) = mapconditions(df_base, no_chips=2, chipspecs='full', save_maps=True, save_pos=True)

    print 'Complete!'
