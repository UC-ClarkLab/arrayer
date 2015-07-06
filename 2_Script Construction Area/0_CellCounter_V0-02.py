
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from skimage.filters import threshold_adaptive, gaussian_filter, threshold_otsu
    from skimage.morphology import disk, binary_opening, watershed, remove_small_objects
    from scipy.ndimage import distance_transform_edt
    from scipy import ndimage
    from skimage.color import label2rgb
    from skimage.measure import regionprops
    from skimage.exposure import equalize_adapthist
    from skimage.feature import peak_local_max
    from scipy.stats import pearsonr
except ImportError:
    raise ImportError("Please verify the package requirements are met.")


# - FUNCTIONS - #


def correct_illumination(grayscale, sigma=400, pickle_me=False):

    '''
    Applies a Gaussian (low pass) filter with a large sigma (default sigma=400)
    to estimate uneven illumination across a grayscale image and correct it.
    This function overcorrects near objects with large image gradients and is
    optimized for use with 16 bit images recorded using a 12 bit camera.

    Inputs:
    -------
    grayscale: A grayscale image loaded into a NumPy array with type np.uint16

    sigma: The standard deviation used for applying the Gaussian filter,
        sigma > 50 is strongly recommended

    pickle_me: Boolean, dumps NumPy arrays of intermediate images to pickle
        files in the current working directory if True

    Outputs:
    --------
    corrected: An illumination corrected NumPy array

    '''
    # sigma = 350 to 400 looks best
    # 65535 is the max value of np.uint16 data type
    background = (gaussian_filter(grayscale, sigma=sigma)*65535).astype(np.uint16)
    inverted = 4096 - background  # inverts light and dark areas
    corrected = (grayscale + inverted)/2

    if pickle_me:
        background.dump('est_background.p')
        inverted.dump('inverted_back.p')
        corrected.dump('corrected.p')

    return(corrected)


def hierarchical_segmentation(grayscale, pickle_me=False):

    '''
    Segments a grayscale image by first applying adaptive histogram
    equalization to enhance contrast followed by an Otsu threshold to
    isolate cell colonies. An adaptive thresholding method is then used to
    isolate clusters of close-packed cells. Estimated regions of interest for
    individual cells are finally generated using the Watershed algorithm, and
    the cell regions are given unique labels and various measurements are
    calculated for the regions from the original grayscale image.

    Inputs:
    -------
    grayscale: A grayscale image loaded into a NumPy array

    pickle_me: Boolean, dumps NumPy arrays of intermediate images to pickle
        files in the current working directory if True

    Outputs:
    --------
    labels: The labels associated with the thresholded regions

    props: The properties of the regions-of-interest measured from the original
        grayscale image
    '''

    # Apply CLAHE
    equalized = equalize_adapthist(grayscale, ntiles_x=16, ntiles_y=16,
                                   clip_limit=0.01, nbins=256)

    # Otsu threshold of CLAHE equalized "grayscale"
    otsu1 = threshold_otsu(equalized)
    print "Otsu threshold: {0}".format(otsu1)

    thresh1 = remove_small_objects(equalized > otsu1)

    colonies = thresh1*equalized

    thresh2 = threshold_adaptive(colonies, 21)
    # Use morphological opening to help separate clusters and remove noise
    opened = binary_opening(thresh2, selem=disk(3))

    clusters = opened*equalized

    # Generate labels for Watershed using local maxima of the distance
    # transform as markers
    distance = distance_transform_edt(opened)
    local_maxi = peak_local_max(distance, min_distance=6,
                                indices=False, labels=opened)
    markers = ndimage.label(local_maxi)[0]
    # plt.imshow(markers)

    # Apply Watershed
    labels = watershed(-distance, markers, mask=opened)
#    plt.imshow(label2rgb(labels))

    # Measure labeled region properties in the illumination-corrected image
    # (not the contrast stretched image)
    props = regionprops(labels, intensity_image=grayscale)

#    fig, axs = plt.subplots(2, 4)
#    axs[0, 0].imshow(equalized, cmap=plt.cm.gray)
#    axs[0, 0].set_title('CLAHE Equalized')
#    axs[0, 1].imshow(thresh1, cmap=plt.cm.gray)
#    axs[0, 1].set_title('Threshold 1, Otsu')
#    axs[0, 2].imshow(colonies, cmap=plt.cm.gray)
#    axs[0, 2].set_title('Colonies')
#    axs[0, 3].imshow(thresh2, cmap=plt.cm.gray)
#    axs[0, 3].set_title('Threshold 2, Adaptive')
#    axs[1, 0].imshow(opened, cmap=plt.cm.gray)
#    axs[1, 0].set_title('Threshold 2, Opened')
#    axs[1, 1].imshow(clusters, cmap=plt.cm.gray)
#    axs[1, 1].set_title('Clusters')
#    axs[1, 2].imshow(distance, cmap=plt.cm.gray)
#    axs[1, 2].set_title('Distance Transform')
#    axs[1, 3].imshow(label2rgb(labels))
#    axs[1, 3].set_title('Labelled Segmentation')

    if pickle_me:
        equalized.dump('CLAHE_equalized.p')
        thresh1.dump('thresh1_otsu.p')
        colonies.dump('colonies.p')
        thresh2.dump('thresh2_adaptive.p')
        opened.dump('opened_thresh2.p')
        clusters.dump('clusters.p')
        distance.dump('distance_transform.p')
        markers.dump('max_dist_markers.p')

    return(labels, props)

######################################################

def count_raw(props):
    counter = 0
    for p in props:
        counter +=1
    print "Cells segmented: {0}".format(counter)
    return counter
#count_raw(props)

def count_filter(props, min_area=36, max_area=144):
    counter = 0
    for p in props:
#        print p.area
        if p.area < min_area:
            cells = 0 # ignore very small items
        elif p.area > max_area*4:
            cells = 0 # ignore very large artifacts
        elif p.area > max_area:
            cells = int(round(p.area/max_area))
        else:
            cells = 1
#        print cells
        counter += cells
    print "Estimated cell count: {0}".format(counter)
    return counter
#count_filter(props)


def flat_nonzero(array):

    '''
    Purges zero-value elements of a NumPy array and returns a flattened
    array of finite numbers. Helpful for histograms and comparing pixel
    intensities for colocalization analysis
    '''

    array = array.astype(np.double)
    array[array == 0] = np.nan
    array = array[np.isfinite(array)]
    return array.flatten()


# - MAIN - #
w1 = plt.imread("20150421-BCP-WTC11-chip2-full_A03_s54_w1_TimePoint_1.TIF")
w2 = plt.imread("20150421-BCP-WTC11-chip2-full_A03_s54_w2_TimePoint_1.TIF")
w3 = plt.imread("20150421-BCP-WTC11-chip2-full_A03_s54_w3_TimePoint_1.TIF")

#w1 = np.load('w1_pickle')
#w2.dump('w2_pickle')
#w3.dump('w3_pickle')

w1c = correct_illumination(w1)
w2c = correct_illumination(w2)


#fig1, axs1 = plt.subplots(2, 2)
#axs1[0, 0].imshow(w1, cmap=plt.cm.gray)
#axs1[0, 0].set_title("DAPI Raw")
#axs1[0, 1].imshow(w1c, cmap=plt.cm.gray)
#axs1[0, 1].set_title("DAPI Corrected")
#axs1[1, 0].imshow(w2, cmap=plt.cm.gray)
#axs1[1, 0].set_title("FITC Raw")
#axs1[1, 1].imshow(w2c, cmap=plt.cm.gray)
#axs1[1, 1].set_title("FITC Corrected")
#fig1.tight_layout()

#fig2, axs2 = plt.subplots(1, 2)
#axs2[0].plot(w1.flatten(), w2.flatten(), 'ro')
#axs2[0].set_title("Raw DAPI vs FITC Pixel Intensities")
#axs2[0].set_xlabel("DAPI Pixel Intensities")
#axs2[0].set_ylabel("FITC Pixel Intensities")
#axs2[1].plot(w1c.flatten(), w2c.flatten(), 'bo')
#axs2[1].set_title("Corrected DAPI vs FITC Pixel Intensities")
#axs2[1].set_xlabel("DAPI Pixel Intensities")
#axs2[1].set_ylabel("FITC Pixel Intensities")
#fig2.tight_layout()


w1labs, w1props = hierarchical_segmentation(w1c)
w2labs, w2props = hierarchical_segmentation(w2c)

bmask = (w1labs != 0)
gmask = (w2labs != 0)
#plt.imshow(gmask*2+bmask, cmap=plt.cm.gray)

w1c_bmask = w1c*bmask
w2c_bmask = w2c*bmask
w1c_gmask = w1c*gmask
w2c_gmask = w2c*gmask



#fig3, axs3 = plt.subplots(1, 2)
#axs3[0].plot(flat_nonzero(w1c_bmask), flat_nonzero(w2c_bmask), 'bo')
#axs3[0].set_title("DAPI vs FITC Intensities, DAPI Mask")
#axs3[0].set_xlabel("DAPI Pixel Intensities")
#axs3[0].set_ylabel("FITC Pixel Intensities")
#axs3[1].plot(flat_nonzero(w1c_gmask), flat_nonzero(w2c_gmask), 'go')
#axs3[1].set_title("DAPI vs FITC Intensities, FITC Mask")
#axs3[1].set_xlabel("DAPI Pixel Intensities")
#axs3[1].set_ylabel("FITC Pixel Intensities")
#fig3.tight_layout()


w1cluster1B = flat_nonzero(w1c*(w1labs == 1))
w2cluster1B = flat_nonzero(w2c*(w1labs == 1))

w1cluster1G = flat_nonzero(w1c*(w2labs == 1))
w2cluster1G = flat_nonzero(w2c*(w2labs == 1))



print pearsonr(w1cluster1B, w2cluster1B)
print pearsonr(w1cluster1G, w2cluster1G)
#fig4, axs4 = plt.subplots(2, 3)
#axs4[0, 0].imshow(w1c*(w1labs == 1), cmap=plt.cm.gray)
#axs4[0, 1].imshow(w2c*(w1labs == 1), cmap=plt.cm.gray)
#axs4[0, 2].plot(w1cluster1B, w2cluster1B, 'bo')
#axs4[0, 2].set_xlabel("DAPI Pixel Intensities")
#axs4[0, 2].set_ylabel("FITC Pixel Intensities")
#axs4[1, 0].imshow(w1c*(w2labs == 1), cmap=plt.cm.gray)
#axs4[1, 1].imshow(w2c*(w2labs == 1), cmap=plt.cm.gray)
#axs4[1, 2].plot(w1cluster1G, w2cluster1G, 'go')
#axs4[1, 2].set_xlabel("DAPI Pixel Intensities")
#axs4[1, 2].set_ylabel("FITC Pixel Intensities")
#fig4.tight_layout()

###################################################
def flat_sum(x):
    '''
    Returns flattened sum of a NumPy array.
    '''
    return np.add.reduce(x.flatten(), dtype=np.float64)

def sum_squares(x):
    '''
    Returns the sum of the element-wise squares of the input.
    '''
    return flat_sum(np.square(x.flatten(), dtype=np.float64))



def mandersr(x, y):
    """
    Calculates a variety of correlation coefficients.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------


    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/stats.py#L2427
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    r_num = flat_sum(x * y)
    r_den = np.sqrt(sum_squares(x) * sum_squares(y))
    r = r_num / r_den
    return r

def overlapk(x, y):
    """
    Calculates a variety of correlation coefficients.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------


    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/stats.py#L2427
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    k_num = flat_sum(x * y)
    k1_den = sum_squares(x)
    k2_den = sum_squares(y)
    k1 = k_num / k1_den
    k2 = k_num / k2_den
    return (k1, k2)

def overlapM(x, y, xmask, ymask):
    """
    Calculates a variety of correlation coefficients.
    NOTE TO SELF: If calculating for individual nuclei, iterate labels
        for the xmask and let the ymask be all labels except the background.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input
    xmask: (N, ) boolean, array_like
        Input
    ymask: (N, ) boolean, array_like
        Input

    Returns
    -------


    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    https://github.com/scipy/scipy/blob/v0.15.1/scipy/stats/stats.py#L2427
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
#    print x.shape, ymask.shape
    M1_num = flat_sum(x*ymask) # s1 = s1i if s2i != 0
    M2_num = flat_sum(y*xmask) # s2 = s2i if s1i != 0
    M1_den = flat_sum(x)
    M2_den = flat_sum(y)
    M1 = M1_num / M1_den
    M2 = M2_num / M2_den
    return (M1, M2)

# see http://www.olympusconfocal.com/java/colocalization/

def calc_PCC_fromlabs(labs, w1, w2):

    '''
    Calculates the Pearson Corelation Coefficient (PCC) between two images
    for each label in a labelled image object.

    Inputs:
    -------
    labs: A labelled image object used to mask individual regions of interest
        for calculation of the PCC in a local area.

    w1: The first image accepted as a NumPy array

    w2: The second image accepted as a NumPy array

    Returns:
    --------
    TO DO
    - return pandas dataframe containing PCC
    - plot a histogram showing the PCC for all labels
    '''
    lablist = []
    corrlist = []
    i = 0
    for i in range(0, len(labs)):
        if (labs == i).any() == False:
            pass # ignore superfluous labels with no objects
        else:
            w1i = flat_nonzero(w1*(labs == i))
            w2i = flat_nonzero(w2*(labs == i))
            corr = pearsonr(w1i, w2i)[0]
            lablist.append(i)
            corrlist.append(pearsonr(w1i, w2i)[0])
            print "Label: {0}, PCC = {1}".format(i, corr)
    series = pd.Series(data=corrlist, index=lablist)
    return series

def calc_coloc_fromlabs(labs, w1, w2, w2mask):

    '''
    Calculates the colocalization coefficients between two images
    for each label in a labelled image object.

    Inputs:
    -------
    labs: A labelled image object used to mask individual regions of interest
        for calculation of the PCC in a local area.

    w1: The first image accepted as a NumPy array

    w2: The second image accepted as a NumPy array

    Returns:
    --------
    TO DO
    - return pandas dataframe containing PCC
    - plot a histogram showing the PCC for all labels
    '''
    print "Attempting to calculate correlation coefficients..."
    labs = np.asarray(labs).flatten()
    w1 = np.asarray(w1, dtype=np.float64).flatten()
    w2 = np.asarray(w2, dtype=np.float64).flatten()
    w2mask = np.asarray(w2mask).flatten()
#    print labs.shape, w1.shape, w2.shape, w2mask.shape
    lablist = []
    corrlist = []
    i = 0
    for i in range(0, max(labs)-1):
        if (labs == i).any() == False:
            print "Label {0} appears to be empty...".format(i)
            pass # ignore superfluous labels with no objects
        else:
            w1i = (w1*(labs == i))
            w2i = (w2*(labs == i))

            pcc = pearsonr(w1i, w2i)[0]
            moc = mandersr(w1i, w2i)
            k1, k2 = overlapk(w1i, w2i)
            M1, M2 = overlapM(w1i, w2i, (labs == i), w2mask)
            coefflist = [pcc, moc, k1, k2, M1, M2]
            lablist.append(i)
            corrlist.append(coefflist)
            print "Label: {0}, [pcc, moc, k1, k2, M1, M2] = {1}".format(i, coefflist)
#    series = pd.Series(data=corrlist, index=lablist)
    df = pd.DataFrame(data=corrlist, index=lablist, columns=["PCC", "MOC", "k1", "k2", "M1", "M2"])
    return df


def scattergram(x, y): # memory intensive...
    # from example: http://matplotlib.org/examples/axes_grid/scatter_hist.html
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axScatter = plt.subplots(figsize=(5.5,5.5))

    # the scatter plot:
    axScatter.scatter(x, y)
    axScatter.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("bottom", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("left", 1.2, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 50, 100])

    #axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 50, 100])

    plt.draw()
    plt.show()
##############################

# from SCIPY 2013 talk
# http://www.youtube.com/watch?v=ar5YtgiXfNI
# https://github.com/mfenner1/py_coloc_utils

import math
import numpy as np
from numpy.core.umath_tests import inner1d
# inner1d computes the inner product on last
# dimension and broadcasts the rest

R, G, B = 0, 1, 2
channelPairs = [(R,G), (R,B), (G,B)]

# safely perform dot product on uint8 arrays
# note the training "." to call sum
def safedot(a, b):
    return(np.multiply(a, b, dtype=np.uint16).
    sum(dtype=np.float64))

# Compute colocalization coefficients on
# the image array
def ccc(ia):
    # means, sumSqMeanErrors are 1x3;
    # others Nx3; indicator is dtype bool;
    # others float64
    sumSqs = \
        inner1d(ia.T, ia.T).astype(np.float64)
    sums = \
        ia.sum(axis=0, dtype=np.float64)
    means = sums / ia.shape[0]
    meanErrors = ia - means

    sqMeanErrors = meanErrors**2
    sumSqMeanErrors = sqMeanErrors.sum(axis=0)
    del sqMeanErrors

    indicator = ia > 0

    # dict of channelPairs -> respective dot product
    crossDot = {(c1, c2) : ia[:,c1][indicator[:,c2]].sum() for c1,c2 in channelPairs}

    results = {}
    for c1, c2 in channelPairs:
        k1 = crossDot[(c1,c2)] / sumSqs[c1]
        k2 = crossDot[(c1,c2)] / sumSqs[c2]

        results[(c1,c2)] = {
            "Pearson" :
                (np.dot(meanErrors[:,c1],
                        meanErrors[:,c2]) /
                np.sqrt(sumSqMeanErrors[c1] *
                        sumSqMeanErrors[c2])),

            "Manders" : math.sqrt(k1*k2),

            "Coloc(m)1" : sumIf[(c1,c2)] / sums[c1],
            "Coloc(m)2" : sumIf[(c2,c1)] / sums[c2],

            "Overlap(k)1" : k1,
            "Overlap(k)2" : k2}

    return results
#####################################
#######################################################

'''
print "Analyzing Correlation for All Stained Nuclei"
series_DAPI = calc_PCC_fromlabs(w1labs, w1, w2)

print "Analyzing Correlation for Oct4 Stained Nuclei"
series_FITC = calc_PCC_fromlabs(w2labs, w1, w2)
   # what to do with areas that are all zeros?

#   calculate PCC and MOC
print "Analyzing Correlation for All Pixels"
totcorr = pearsonr(w1c.flatten(), w2c.flatten())[0]

print "Results Summary:"
print "Average Pearson Correlation for ALL Stained Nuclei:"
print "{0} +/- {1}".format(series_DAPI.mean(), series_DAPI.std())
print "Average Pearson Correlation for Oct4 Stained Nuclei:"
print "{0} +/- {1}".format(series_FITC.mean(), series_FITC.std())
print "Pearson Correlation for the Whole Images:"
print "{0}".format(totcorr)

fig5, axs5 = plt.subplots(1, 2)
axs5[0].hist(series_DAPI, bins=1000)
axs5[0].set_title("Pearson Correlation for Hoescht-Stained Nuclei")
axs5[1].hist(series_FITC, bins=1000)
axs5[1].set_title("Pearson Correlation for Oct4-Stained Nuclei")
'''
#df_DAPI = calc_coloc_fromlabs(w1labs, w1c, w2c, (w2labs != 0))
#df_FITC = calc_coloc_fromlabs(w2labs, w2c, w1c, (w1labs != 0))


# Using Pandas
# df = pd.DataFrame(index=['hi', 'yo', 'hey'], columns=['one', 'two', 'three'], data=np.eye(3, 3))
# access column "two" -> df['two']
# access row label "yo" -> df.loc['yo']
# access row index 3 -> df.iloc[3]

#scattergram(x, y)
M1, M2 = overlapM(w1c, w2c, (w1labs == 1), (w2labs != 0))
print M1, M2
plt.show()
print "Complete!"
# NOTES TO SELF:
# APPLY ILLUMINATION CORRECTION BEFORE THRESHOLDING!!!
# So far (as of 2015/04/28) best solution appears to involve applying at least 2 thresholds in series
# first threshold isolates colonies/isolated clusters
# second threshold isolates densely packed clusters
# CLAHE appears to offer the best contrast enhancement solution for thresholding
# UP NEXT! Determine best parameters for CLAHE
# ON DECK! Try a variety of thresholding solutions with CLAHE!!!

# OTHER OPTIONS:
# Might also consider leveraging transmitted light images to identify cell boundaries/edges

# The Sobel filter looks best - may help distinguish cell boundaries
# Try Gaussian filter with sigma ~ 2 before Sobel
# Try remove_small_objects after Sobel
# There are essentially NO differences between the
# skeletonized Sobel filter from the RAW image and the one from the CORRECTED image



# Now I need to:
# IMPLEMENT QUANTITATIVE COLOCALIZATION ANALYSIS
# Plot Blue pixel intensities vs Green pixel intensities
# across the whole image and for single cells (or small clusters)
# Leverage the opened nuclear mask I generated here
# try applying the "opened" mask to the green channel
# - count cells with co-staining and without
# - may need to generate a mask for the green channel


# Also need to purge the unnecessary code from this file
# and make this file run-able/importable as a function