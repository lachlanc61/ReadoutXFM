import numpy as np
import os
import sys
import pybaselines.smooth

#from colorsys import hsv_to_rgb

import config
import src.utils as utils
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#-----------------------------------
#MODIFIABLE CONSTANTS
#-----------------------------------

MIN_E=1.04      #minimum energy of interest
MIN_XE=-5       #extended minimum x for ir
ELASTIC=17.44   #energy of tube Ka
EOFFSET=3.0
MAX_E=30        #maximum energy of interest
SDS=9           #standard deviations
RGBLOG=True     #map RGB as log of intensity
NCOLS=5         #no. colours


#-----------------------------------
#INITIALISE
#-----------------------------------

# create a pointer to the module object instance itself
#       functions like "self" for module
#   https://stackoverflow.com/questions/1977362/how-to-create-module-wide-variables-in-python
this = sys.modules[__name__]

#vars for gaussians
#   x-zero
xzer=np.floor(-(MIN_XE/config.ESTEP)).astype(int)   
#   standard deviation 
sd=(ELASTIC-MIN_E)/(SDS)  
#   means for each
irmu=MIN_E-sd*1.5   #ir
rmu=MIN_E+sd*1.5    #red
gmu=rmu+sd*3        #green
bmu=ELASTIC-sd*1.5    #blue
uvmu=ELASTIC+sd*1.5   #uv

#-----------------------------------
#FUNCTIONS
#-----------------------------------

def initialise(e):
    """
    initialise the colour gaussians 
    export to module-wide variables via "this"

    receives energy channel list
    returns None

    
    NB: not certain this is optimal - just need to get e somehow
        could also create in parallel via config.NCHAN & ESTEP
    """
    kachan=utils.lookfor(e,float(ELASTIC+EOFFSET))  #K-alpha channel
    npad=config.NCHAN-kachan

    hsv=cm.get_cmap('hsv', kachan)
    cmap=hsv(range(kachan))

    cred=cmap[:,0]
    cgreen=cmap[:,1]
    cblue=cmap[:,2]

    cred=np.pad(cred, (0, npad), 'constant', constant_values=(0, 1))
    cgreen=np.pad(cgreen, (0, npad), 'constant', constant_values=(0, 0))
    cblue=np.pad(cblue, (0, npad), 'constant', constant_values=(0, 0))    

    #assign to module-wide variables
    this.red=cred
    this.green=cgreen
    this.blue=cblue

    return None

def spectorgb(e, y):
    """
    maps spectrum onto R G B channels 
    use HSV colourmap to generate
    """
    #bg = pybaselines.smooth.snip(y,SNIPWINDOW)[0]
    #y=y-bg

    #if doing log y
    if RGBLOG:
        #convert y to float for log
        yf=y.astype(float)
        #log y, excluding 0 values
        y=np.log(yf, out=np.zeros_like(yf), where=(yf!=0))

    #multiply y vectorwise onto channels (t/px: 0.004051 s)
    rsum=np.sum(y*(this.red))/len(e)
    gsum=np.sum(y*(this.green))/len(e)
    bsum=np.sum(y*(this.blue))/len(e)

    ysum=np.sum(y)
    
#    max=np.max([rsum,bsum,gsum])

    return(rsum,gsum,bsum,ysum)



def clcomplete(rvals, gvals, bvals, mapx, mapy):
    """
    creates final colour-mapped image

    recives R G B arrays per pixel, and total counts per pixel

    displays plot
    """
    print(f'rgb maxima: r {np.max(rvals)} g {np.max(gvals)} b {np.max(bvals)}')
    chmax=np.max([np.max(rvals),np.max(gvals),np.max(bvals)])
    rvals=rvals/chmax
    gvals=gvals/chmax
    bvals=bvals/chmax

    print(f'scaled maxima: r {np.max(rvals)} g {np.max(gvals)} b {np.max(bvals)}')

    np.savetxt(os.path.join(config.odir, "rvals.txt"), rvals)
    np.savetxt(os.path.join(config.odir, "gvals.txt"), gvals)
    np.savetxt(os.path.join(config.odir, "bvals.txt"), bvals)

    rimg=np.reshape(rvals, (-1, mapx))
    gimg=np.reshape(gvals, (-1, mapx))
    bimg=np.reshape(bvals, (-1, mapx))

    rgbarray = np.zeros((mapy,mapx,3), 'uint8')
    rgbarray[..., 0] = rimg*256
    rgbarray[..., 1] = gimg*256
    rgbarray[..., 2] = bimg*256
    
    return(rgbarray)

def clcomplete2(rvals, gvals, bvals, totalcounts, mapx, mapy):
    """
    creates final colour-mapped image

    recives R G B arrays per pixel, and total counts per pixel

    displays plot
    """
    totalpx = len(totalcounts)
    
    print(f'rgb maxima: r {np.max(rvals)} g {np.max(gvals)} b {np.max(bvals)}')
    allch=np.append(rvals,gvals)   
    allch=np.append(allch,bvals)  
    chmax=max(allch)

    maxcounts=max(totalcounts)

    for i in np.arange(totalpx):
        rgbscale=totalcounts[i]/maxcounts
        rvals[i]=rvals[i]*rgbscale/chmax
        gvals[i]=gvals[i]*rgbscale/chmax
        bvals[i]=bvals[i]*rgbscale/chmax

    print(f'scaled maxima: r {np.max(rvals)} g {np.max(gvals)} b {np.max(bvals)}')

    np.savetxt(os.path.join(config.odir, "rvals.txt"), rvals)
    np.savetxt(os.path.join(config.odir, "gvals.txt"), gvals)
    np.savetxt(os.path.join(config.odir, "bvals.txt"), bvals)

    rreshape=np.reshape(rvals, (-1, mapx))
    greshape=np.reshape(gvals, (-1, mapx))
    breshape=np.reshape(bvals, (-1, mapx))

    rgbarray = np.zeros((mapy,mapx,3), 'uint8')
    rgbarray[..., 0] = rreshape*256
    rgbarray[..., 1] = greshape*256
    rgbarray[..., 2] = breshape*256
    
    return(rgbarray)


def clshow(rgbarray):
    plt.imshow(rgbarray)
    plt.savefig(os.path.join(config.odir, 'colours.png'), dpi=150)
    plt.show()   


"""
speedup:    
    for j:                  0.007625 s
    vectorise channels:     0.004051 s
    pre-init gaussians:     0.002641 s   
    colourmap:              0.001886 s
    fit snip:               0.002734 s
    fit complex snip:       0.002919 s
"""