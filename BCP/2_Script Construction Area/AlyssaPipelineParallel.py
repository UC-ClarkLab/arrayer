# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:39:29 2015

@author: Brian Perea
"""
try:
    import ArrayerAnalysis as ogre
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.color import label2rgb
    import seaborn as sns
    import pandas as pd
    import os
    from joblib import Parallel, delayed
    import multiprocessing
except ImportError:
    raise ImportError("Please verify the module is in the working directory.")

#def ogre_show(axrow, x, y, z):
#    axrow[0].imshow(x, cmap=plt.cm.gray)
#    axrow[1].imshow(y, cmap=plt.cm.gray)
#    axrow[2].imshow(z, cmap=plt.cm.gray)
    


def liveDead(row, f): 
    path, name = os.path.split(f)
    
    img = ogre.site_to_array(f)
    
    b = img[:, :, 0] # DAPI
    g = img[:, :, 1] # FITC
    r = img[:, :, 2] # TexasRed
    
    bc = ogre.correct_illumination(b)
    bcmask = ogre.adapt_thresh(bc)
    blabs, bprops = ogre.watershed_label(bcmask, bc)
    
    gc = ogre.correct_illumination(g)
    gcmask = ogre.adapt_thresh(gc)
    glabs, gprops = ogre.watershed_label(gcmask, gc)
    
    rc = ogre.correct_illumination(r)
    rcmask = ogre.adapt_thresh(rc)
    rlabs, rprops = ogre.watershed_label(rcmask, rc)
    
#    fig, ax = plt.subplots(3,3, figsize=(15,12))
#    fig.suptitle(f, fontsize=15)
#    ax[0,0].imshow(bc, cmap=plt.cm.gray)
#    ax[0,1].imshow(gc, cmap=plt.cm.gray)
#    ax[0,2].imshow(rc, cmap=plt.cm.gray)
#    ax[1,0].imshow(bcmask, cmap=plt.cm.gray)
#    ax[1,1].imshow(gcmask, cmap=plt.cm.gray)
#    ax[1,2].imshow(rcmask, cmap=plt.cm.gray)
#    ax[2,0].imshow(label2rgb(blabs))
#    ax[2,1].imshow(label2rgb(glabs))
#    ax[2,2].imshow(label2rgb(rlabs))
#    sns.despine()
#    fig.tight_layout()
    
    # Raw cell counter seems to be doing a better job here...
    raw_dapi = ogre.count_raw(bprops)
#    bcount_filter = ogre.count_filter(bprops)
    raw_fitc = ogre.count_raw(gprops)
#    gcount_filter = ogre.count_filter(gprops)
    raw_texas = ogre.count_raw(rprops)
#    rcount_filter = ogre.count_filter(rprops)
    
    glabmask = glabs != 0
    rlabmask = rlabs != 0

    live = 0
    dead = 0
    dying = 0
    for i in range(1, blabs.max()):
        nucleus_roi = blabs == i
        if((nucleus_roi * glabmask * rlabmask).any()):
            dying += 1
        elif((nucleus_roi*glabmask).any()):
            live += 1
        elif((nucleus_roi * rlabmask).any()):
            dead += 1
    
    return (row, [path, name, raw_dapi, raw_fitc, raw_texas, live, dead, dying])




if __name__ == '__main__':
    # - MAIN - #
    platepath = 'B:\\Projects\\Research\\2015-05-11 AlyssaScreen1\\20150511-AlyssaC2D-D3-LD_Plate_4714'
    output_filename = '20150511-AlyssaC2D-D3-LD_Plate_4714.csv'
    
    output_filepath = os.path.join(platepath, output_filename)
    w1_files = ogre.iter_plate(platepath)
    w1_files_length = len(w1_files)
    
    columns = ["path", "file", "raw_dapi", "raw_fitc", "raw_texas", "live", "dead", "all"]
    df = pd.DataFrame(index=range(0, w1_files_length), columns=columns)
    
    # Parallel for loop equivalent to "for f in w1_files:"
    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores, verbose=11)(delayed(liveDead)(row, f) for row, f in zip(range(w1_files_length), w1_files))  
    for result in results:
        df.iloc[result[0]] = result[1]

    df.to_csv(output_filepath)


    print "Complete!"