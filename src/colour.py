import numpy as np
import os
import sys

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

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def initialise2(e):
    """
    initialise the colour gaussians 
    export to module-wide variables via "this"

    receives energy channel list
    returns None

    
    NB: not certain this is optimal - just need to get e somehow
        could also create in parallel via config.NCHAN & ESTEP
    """
    #extend x-space to negative values 
    #   (needed to normalise ir gaussian)
    xe=np.arange(-5,0,config.ESTEP)
    xe=np.append(xe,e)

    #create ir gaussian, then truncate back
    irgauss=utils.normgauss(xe, irmu, sd, 1)
    irgauss=irgauss[xzer:]

    #create other gaussians
    #   normalised to 1
    #   note: e, rmu etc = module-level variables created on import
    rgauss=utils.normgauss(e, rmu, sd, 1)
    ggauss=utils.normgauss(e, gmu, sd, 1)
    bgauss=utils.normgauss(e, bmu, sd, 1)
    uvgauss=utils.normgauss(e, uvmu, sd, 1)

    #combine into colour assignment channels
    red=rgauss+uvgauss
    green=ggauss
    blue=bgauss+irgauss

    #normalise so that sum(red,green,blue) always = 1.0
    mult=np.divide(1,red+green+blue)
    red=red*mult
    green=green*mult
    blue=blue*mult

    #assign to module-wide variables
    this.red=red
    this.green=green
    this.blue=blue

    return None


def initialise(e):
    """
    initialise the colour gaussians 
    export to module-wide variables via "this"

    receives energy channel list
    returns None

    
    NB: not certain this is optimal - just need to get e somehow
        could also create in parallel via config.NCHAN & ESTEP
    """
    kachan=utils.lookfor(e,float(ELASTIC))  #K-alpha channel
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
    maps spectrum onto R G B channels weighted by series of gaussians

        R G B gaussians at ~1/3 2/3 3/3 across region of interest

        + two extended gaussians to cause purple shift at low and high 
            "ir"(coloured blue) and "uv"(coloured red)

        not properly linear, peaks halfway between gaussians currently weighted ~20% lower than centres

        ATTN: reads local constants and initialised vars eg. RGBLOG, xzer


        speedup:    
            for j:                  0.007625 s
            vectorise channels:     0.004051 s
            pre-init gaussians:     0.002641 s    
    """
    #if doing log y
    if RGBLOG:
        #convert y to float for log
        yf=y.astype(float)
        #log y, excluding 0 values
        y=np.log(yf, out=np.zeros_like(yf), where=(yf!=0))

    #multiply y vectorwise onto channels (t/px: 0.004051 s)
    rsum=np.sum(y*(this.red)*max(y))/len(e)
    gsum=np.sum(y*(this.green)*max(y))/len(e)
    bsum=np.sum(y*(this.blue)*max(y))/len(e)

    ysum=np.sum(y)
    
    return(rsum,gsum,bsum,ysum)



def clcomplete(rvals, gvals, bvals, totalcounts, mapx, mapy):
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

