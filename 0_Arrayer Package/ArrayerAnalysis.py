
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
    from scipy.stats import pearsonr, mode#, pointbiserialr
    from scipy import special
    # pointbiserialr used to measure the relationship between a 
    # binary variable, x, and a continuous variable, y
    import seaborn as sns
    import h5py
    import os
    import re

except ImportError:
    raise ImportError("Please verify the package requirements are met.")

sns.set(context="talk", style="white", palette="gray")
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

def clahe_adapthist(grayscale):
    # Apply CLAHE
    equalized = equalize_adapthist(grayscale, ntiles_x=16, ntiles_y=16,
                                   clip_limit=0.01, nbins=256)
    return(equalized)
    
def adapt_thresh(equalized, returnmasked=False):
    # Normalize by subtracting image mean
    # Clip values less than zero to remove background signal
    equalized = (equalized - equalized.mean()).clip(min=0)
    
    # Otsu threshold of CLAHE equalized "grayscale"
    otsu1 = threshold_otsu(equalized)
#    print "Otsu threshold: {0}".format(otsu1)

    thresh1 = remove_small_objects(equalized > otsu1)

    colonies = thresh1*equalized

    thresh2 = threshold_adaptive(colonies, 21)
    # Use morphological opening to help separate clusters and remove noise
    opened = binary_opening(thresh2, selem=disk(3))
    if returnmasked == True:
        clusters = opened*equalized
        return(opened, masked)
    else:
        return opened
        
def watershed_label(opened, grayscale):
    '''Accepts True/False mask "opened" and intensity image "grayscale".
    Returns labelled image regions and properties.
    '''
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
    return(labels, props)
    
######################################################

def count_raw(props):
    counter = 0
    for p in props:
        counter +=1
#    print "Cells segmented: {0}".format(counter)
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
#    print "Estimated cell count: {0}".format(counter)
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

def hdf5ify(target, dest):
    '''Work in progress...'''    
    # target must be path to target directory
    # dest must be path to destination hdf5 file (will be created if not already existing)
        
    hdf5 = h5py.File(dest, 'a')
    
    # Make a list of all subdirectories (time point folders)
    subdirectories = os.walk('.').next()[1]
        
    # Iterate through the subdirectories
    # Identify metadata in the file name/directory
    # Create the relevant folder hierarchy for sorting
    # Rename and move the file in one step
    for item in subdirectories:
        # Identify the time point metadata from the file directory
        timepoint = re.search('TimePoint_\d{1,3}', item)
        timepoint = timepoint.group(0)
        path = os.path.join(target, timepoint)
        # Create a list of files in the current subdirectory
        files = os.listdir(path)	
        # In that list of files, iterate file-by-file and take the relevant actions
        for f in files:
            # Split file name from the extension 				
#            name, ext = os.path.splitext(f)
            #print name
#            # Old file name (including full path)
#            old = directory + '/' + timepoint + '/' + name + ext
            # Check for thumbnails
            # Identify thumbnail metadata
            thumbcheck = re.search('Thumb', f)
            # If file is a thumbnail, do nothing
            if (thumbcheck):
                # Ignore thumbnails...
                pass
            else:		
                # Identify well metadata
                wellcheck = re.search('A\d{2}', f)
                # Identify site metadata
                sitecheck = re.search('s\d{1,4}', f)
                # Identify channel (wavelength) metadata
                w1check = re.search('w1', f)

                # Read images to NumPy arrays
                # Stack images from same site AND well
                if(w1check):
                    
                    sarray = plt.imread(os.path.join(path, f))
                    
                    w2check = f.replace('w1', 'w2')
                    w3check = f.replace('w1', 'w3')
                    w4check = f.replace('w1', 'w4')

                    if(w2check in files):
                        w2 = plt.imread(os.path.join(path, w2check))
                        np.dstack(sarray, w2)
                        
                    if(w3check in files):
                        w3 = plt.imread(os.path.join(path, w3check))
                        np.dstack(sarray, w3)
                        
                    if(w4check in files):
                        w4 = plt.imread(os.path.join(path, w4check))
                        np.dstack(sarray, w4)
                    
                      
                    
                
                        
                '''
                if (wavecheck):
                    # Provide status indicators for the user
                    print 'Image identified!'
                    # Indicate which aspects of the image metadata have been identified and what their values are
                    print 'Channel: ' + wavecheck.group()				
                    # If the relevant site folder does not exist, create it
                    if os.path.isdir(directory + '/' + wavecheck.group()) == False:
                        os.makedirs(directory + '/' + wavecheck.group())									
                        # Simultaneously move and rename the file by modifying its full path
                        os.rename(old, directory + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext)
                        # Print output file location
                        print directory + '/' + wavecheck.group() + '/' + name + '_' + timepoint + ext									
            # Clean up
            shutil.rmtree((directory + '/' + timepoint))
            print 'Complete!'
            '''

    hdf5.close()

    return

def iter_plate(platepath):
    '''Iterate through plate folder, extracting the absolute paths
    of image files from wavelength 1 (w1) for all timepoints.
    '''
    w1_files = []
    for root, dirs, files in os.walk(platepath):

        timecheck = re.search('TimePoint_\d{1,3}', root)

        if(timecheck):
            for f in files:
                w1check = re.search('w1', f)
                if(w1check):
                    w1_files.append(os.path.join(root, f))

            
    return(w1_files)

def site_to_array(sitepath):
    '''Accepts absolute path to a single w1 image file (wavelength 1).
    Checks for additional channels and compiles all channels into a
    single three dimensional NumPy array.
    '''
    path, name = os.path.split(sitepath)
    
    w1check = re.search('w1', name)
    if(w1check):
        sarray = plt.imread(sitepath)

        files = os.listdir(path)
        w2check = name.replace('w1', 'w2')
        w3check = name.replace('w1', 'w3')
        w4check = name.replace('w1', 'w4')
        if(w2check in files):
            w2 = plt.imread(os.path.join(path, w2check))
            sarray = np.dstack((sarray, w2))

        if(w3check in files):
            w3 = plt.imread(os.path.join(path, w3check))
            sarray = np.dstack((sarray, w3))

        if(w4check in files):
            w4 = plt.imread(os.path.join(path, w4check))
            sarray = np.dstack((sarray, w4))

    return(sarray)
    

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

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis

def betai(a, b, x):
    """
    Returns the incomplete beta function.

    I_x(a,b) = 1/B(a,b)*(Integral(0,x) of t^(a-1)(1-t)^(b-1) dt)

    where a,b>0 and B(a,b) = G(a)*G(b)/(G(a+b)) where G(a) is the gamma
    function of a.

    The standard broadcasting rules apply to a, b, and x.

    Parameters
    ----------
    a : array_like or float > 0

    b : array_like or float > 0

    x : array_like or float
        x will be clipped to be no greater than 1.0 .

    Returns
    -------
    betai : ndarray
        Incomplete beta function.

    """
    x = np.asarray(x)
    x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    #return special.betainc(a, b, x)
    return special.betainc(a, b, x)

def ss(a, axis=0):
    """
    Squares each element of the input array, and returns the sum(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        The axis along which to calculate. If None, use whole array.
        Default is 0, i.e. along the first axis.

    Returns
    -------
    ss : ndarray
        The sum along the given axis for (a**2).

    See also
    --------
    square_of_sums : The square(s) of the sum(s) (the opposite of `ss`).

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([1., 2., 5.])
    >>> stats.ss(a)
    30.0

    And calculating along an axis:

    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> stats.ss(b, axis=1)
    array([ 30., 65.])

    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def pearsonr(x, y):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)

    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(ss(xm) * ss(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n-2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        prob = betai(0.5*df, 0.5, df / (df + t_squared))
    return r, prob

def pearsonr2(x, y):
    """
    Calculates the Pearson correlation coefficient.

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
    # x and y should have same length.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(ss(xm) * ss(ym))
    r = r_num / r_den    
    
    r_num = flat_sum(x - x.mean()) * flat_sum(y - y.mean())
    r_den = np.sqrt(flat_sum(np.square(x - x.mean(), dtype=np.float64)) * flat_sum(np.square(y - y.mean(), dtype=np.float64)))
    print r_num, r_den
    r = r_num / r_den
    return r

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
    '''
    Requires multichannel image (ia) as input.
    '''
    
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
