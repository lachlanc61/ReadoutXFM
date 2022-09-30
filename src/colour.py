import numpy as np
import os
import sys

#from colorsys import hsv_to_rgb

import config
import src.utils as utils
import matplotlib.pyplot as plt

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

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

#vars for gaussians
#   x-zero
xzer=np.floor(-(MIN_XE/config.ESTEP)).astype(int)   
#   standard deviation 
sd=(MAX_E-MIN_E)/(SDS)  
#   means for each
irmu=MIN_E-sd*1.5   #ir
rmu=MIN_E+sd*1.5    #red
gmu=rmu+sd*3        #green
bmu=MAX_E-sd*1.5    #blue
uvmu=MAX_E+sd*1.5   #uv


#-------------------------------------
#FUNCTIONS
#-----------------------------------

"""
def initialise(e):

    initialise the colour gaussians as module-wide variables


    #extend x-space to negative values 
    #   (needed to normalise ir gaussian)
    xe=np.arange(-5,0,config.ESTEP)
    xe=np.append(xe,e)

    #create ir gaussian, then truncate back
    ir=utils.normgauss(xe, irmu, sd, max(y))
    ir=ir[xzer:]

    #create other gaussians
    #   normalised to max(y)
    #   note: e, rmu etc = module-level variables created on import
    red=utils.normgauss(e, rmu, sd, max(y))
    green=utils.normgauss(e, gmu, sd, max(y))
    blue=utils.normgauss(e, bmu, sd, max(y))
    uv=utils.normgauss(e, uvmu, sd, max(y))
"""

def spectorgb(e, y):
    """
    maps spectrum onto R G B channels weighted by series of gaussians

        R G B gaussians at ~1/3 2/3 3/3 across region of interest

        + two extended gaussians to cause purple shift at low and high 
            "ir"(coloured blue) and "uv"(coloured red)

        not properly linear, peaks halfway between gaussians currently weighted ~20% lower than centres

        ATTN: reads local constants and initialised vars eg. RGBLOG, xzer
    """
    #if doing log y
    if RGBLOG:
        #convert y to float for log
        yf=y.astype(float)
        #log y, excluding 0 values
        y=np.log(yf, out=np.zeros_like(yf), where=(yf!=0))

    #extend x-space to negative values 
    #   (needed to normalise ir gaussian)
    xe=np.arange(-5,0,config.ESTEP)
    xe=np.append(xe,e)

    #create ir gaussian, then truncate back
    ir=utils.normgauss(xe, irmu, sd, max(y))
    ir=ir[xzer:]

    #create other gaussians
    #   normalised to max(y)
    #   note: e, rmu etc = module-level variables created on import
    red=utils.normgauss(e, rmu, sd, max(y))
    green=utils.normgauss(e, gmu, sd, max(y))
    blue=utils.normgauss(e, bmu, sd, max(y))
    uv=utils.normgauss(e, uvmu, sd, max(y))

    #multiply y vectorwise onto channels (t/px: 0.004051 s)
    rsum=np.sum(y*(red+uv))/len(e)
    gsum=np.sum(y*(green))/len(e)
    bsum=np.sum(y*(blue+ir))/len(e)

    yret=np.sum(y)
    
    return(rsum,gsum,bsum,yret)



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
    plt.show()   

