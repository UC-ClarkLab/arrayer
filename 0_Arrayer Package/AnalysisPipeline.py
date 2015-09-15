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
except ImportError:
    raise ImportError("Please verify the module is in the working directory.")
#%%
#def ogre_show(axrow, x, y, z):
#    axrow[0].imshow(x, cmap=plt.cm.gray)
#    axrow[1].imshow(y, cmap=plt.cm.gray)
#    axrow[2].imshow(z, cmap=plt.cm.gray)

# - MAIN - #
location = 'home'

if location == 'home':
    platepath = 'B:\\Projects\\Research\\2015-05-11 AlyssaScreen1'
    outputpath = 'B:\\ownCloud\\0_Programming\\Research\\Image Processing\\Arrayer Package'
else:
    platepath = 'C:\Users\Brian\Desktop\Data For Analysis\2015-05-11 AlyssaScreen1'
    outputpath = 'C:\Users\Brian\Documents\GitHub\clarklab.darkside\0_Arrayer Package'


output_filename = '2015-05-11 AlyssaScreen1.csv'
output_filepath = os.path.join(outputpath, output_filename)
w1_files = ogre.iter_plate(platepath)
w1_files_length = len(w1_files)

columns = ["path", "file", "raw_dapi", "raw_fitc", "raw_texas", "live", "dead", "all"]
df = pd.DataFrame(index=range(0, w1_files_length), columns=columns)
#%%
for f in w1_files:
    row = w1_files.index(f)
    path, name = os.path.split(f)

    print "Status Update:"
    print "File: {0}".format(f)
    print "Item: {0}/{1}".format(row, w1_files_length)

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

    break
#%%
    bmask = (blabs != 0)
    gmask = (glabs != 0)
    rmask = (rlabs != 0)
    #plt.imshow(gmask*2+bmask, cmap=plt.cm.gray)

    dapi = bc*bmask
    fitc = gc*gmask
    texas = rc*rmask

#%%

    sel = 10

    plive = ogre.pearsonr((dapi*(blabs == sel)).flatten(), fitc.flatten())
    pdead = ogre.pearsonr((dapi*(blabs == sel)).flatten(), texas.flatten())
    print 'plive = {0}, pdead = {1}'.format(plive, pdead)

    plive2 = ogre.pearsonr2(dapi*(blabs == sel), fitc)
    pdead2 = ogre.pearsonr2(dapi*(blabs == sel), texas)
    print 'plive2 = {0}, pdead2 = {1}'.format(plive2, pdead2)

    mlive = ogre.mandersr(dapi*(blabs == sel), fitc)
    mdead = ogre.mandersr(dapi*(blabs == sel), texas)
    print 'mlive = {0}, mdead = {1}'.format(mlive, mdead)

    klive = ogre.overlapk(dapi*(blabs == sel), fitc)
    kdead = ogre.overlapk(dapi*(blabs == sel), texas)
    print 'klive = {0}, kdead = {1}'.format(klive, kdead)

    Mlive = ogre.overlapM(dapi*(blabs == sel), fitc, bmask, gmask)
    Mdead = ogre.overlapM(dapi*(blabs == sel), texas, bmask, rmask)
    print 'Mlive = {0}, Mdead = {1}'.format(Mlive, Mdead)

#    pearson = ogre.calc_PCC_fromlabs(blabs, bc, gc)
#    print pearson

#%%
    coef = ogre.calc_coloc_fromlabs(blabs, bc, gc, gmask)
#%%
    from scipy.stats import pearsonr



#%%


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

'''
    # Raw cell counter seems to be doing a better job here...
    raw_dapi = ogre.count_raw(bprops)
#    bcount_filter = ogre.count_filter(bprops)
    raw_fitc = ogre.count_raw(gprops)
#    gcount_filter = ogre.count_filter(gprops)
    raw_texas = ogre.count_raw(rprops)
#    rcount_filter = ogre.count_filter(rprops)

    print "Raw Count DAPI: {0}".format(raw_dapi)
    print "Raw Count FITC: {0}".format(raw_fitc)
    print "Raw Count Texas: {0}".format(raw_texas)

    glabmask = glabs != 0
    rlabmask = rlabs != 0


    live = 0
    dead = 0
    dying = 0
    # for i in range(1, blabs.max()):
    for i in range(1, 2):
        nucleus_roi = blabs == i

        # Insert statistical calculations here

        # Calculate correlation coefficients
        # Hypothesis test to determine whether to keep or reject




        # if((nucleus_roi*glabmask).any()):
            # live += 1
        # if((nucleus_roi * rlabmask).any()):
            # dead += 1
        # if((nucleus_roi * glabmask * rlabmask).any()):
            # dying += 1
    # print "Count Nuclear and Live: {0}".format(live)
    # print "Count Nuclear and Dead: {0}".format(dead)
    # print "Count Nuclear and Live and Dead: {0}".format(dying)

    df.iloc[row] = [path, name, raw_dapi, raw_fitc, raw_texas, live, dead, dying]



    break


df.to_csv(output_filepath)
'''


'''
#w1 = plt.imread("20150421-BCP-WTC11-chip2-full_A03_s54_w1_TimePoint_1.TIF")
#w2 = plt.imread("20150421-BCP-WTC11-chip2-full_A03_s54_w2_TimePoint_1.TIF")
#w3 = plt.imread("20150421-BCP-WTC11-chip2-full_A03_s54_w3_TimePoint_1.TIF")

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
max_mask = (bmask | gmask)
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
sg = sns.jointplot(flat_nonzero(w1c*max_mask), flat_nonzero(w2c*max_mask))
sg.set_axis_labels(xlabel="DAPI", ylabel="FITC")
sns.puppyplot()
'''
print "Complete!"