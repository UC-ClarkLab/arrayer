import numpy as np
import h5py

# To open a database:
# f = h5py.File('databaseName.h5', 'r')

# For CellProfiler index hierarchy:
# f['Measurements'][f['Measurements'].keys()[0]] gets you to the top level, e.g.
# f = f['Measurements'][f['Measurements'].keys()[0]]
# Then can use selectors for the data needed. i.e. anything in f.keys()
# f[DataGroup][Feature][index]