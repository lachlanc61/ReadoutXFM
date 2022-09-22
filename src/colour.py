import numpy as np
import os

from colorsys import hsv_to_rgb

import config
import src.utils as utils
import matplotlib.pyplot as plt

#-----------------------------------
#MODIFIABLE VARS
#-----------------------------------

#colour-related variables
mine=1.04   #minimum energy of interest
minxe=-5    #extended minimum x for ir
elastic=17.44  #energy of tube Ka
maxe=30  #maximum energy of interest
sds=9   #standard deviations
rgblogscale=True    #map RGB as log of intensity
ncols=5

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def spectorgb(e, y):
    """
    map spectrum onto R G B channels weighted by series of gaussians

        R G B gaussians at ~1/3 2/3 3/3 across region of interest
        + two "extended" gaussians at extremes, "ir"(=blue) and "uv"(=red)

        not properly linear, peaks halfway between gaussians currently weighted ~20% lower than centres
    """
    if rgblogscale:
        #convert y to float for log
        yf=y.astype(float)
        #log y, excluding 0 values (ie. 0 stays 0)
        y=np.log(yf, out=np.zeros_like(yf), where=(yf!=0))

    #max of ir curve is outside e
    #   need to extend x to -5 to normalise correctly
    xe=np.arange(-5,0,config.ESTEP)
    xe=np.append(xe,e)
    
    #create ir gaussian, then truncate back
    ir=utils.normgauss(xe, irmu, sd, max(y))
    ir=ir[xzer:]

    #create other gaussians
    #   currently depends on a lot of variables outside function
    #   should spin this off into own script and put variable definitions there
    red=utils.normgauss(e, rmu, sd, max(y))
    green=utils.normgauss(e, gmu, sd, max(y))
    blue=utils.normgauss(e, bmu, sd, max(y))
    uv=utils.normgauss(e, uvmu, sd, max(y))

    #initialise channel outputs
    rch=np.zeros(len(e))
    gch=np.zeros(len(e))
    bch=np.zeros(len(e))

    #calculate RGB matrices
    #step through spectrum by energy j
    #   multiplying y by gaussian value at that j
    for j in np.arange(len(e)):
        rch[j]=y[j]*(red[j]+uv[j])
        gch[j]=y[j]*(green[j])
        bch[j]=y[j]*(blue[j]+ir[j])
    #calculate average per channel
    rret=np.sum(rch)/len(e)
    gret=np.sum(gch)/len(e)
    bret=np.sum(bch)/len(e)
    yret=np.sum(y)
    
    return(rret,gret,bret,yret)



def clcomplete(rvals, gvals, bvals, totalcounts):
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

    rreshape=np.reshape(rvals, (-1, config.MAPX))
    greshape=np.reshape(gvals, (-1, config.MAPX))
    breshape=np.reshape(bvals, (-1, config.MAPX))

    rgbarray = np.zeros((config.MAPY,config.MAPX,3), 'uint8')
    rgbarray[..., 0] = rreshape*256
    rgbarray[..., 1] = greshape*256
    rgbarray[..., 2] = breshape*256
    
    return(rgbarray)

def clshow(rgbarray):
    plt.imshow(rgbarray)
    plt.show()   

#-----------------------------------
#INITIALISE
#-----------------------------------
xzer=np.floor(-(minxe/config.ESTEP)).astype(int)
sd=(maxe-mine)/(sds)
irmu=mine-sd*1.5
rmu=mine+sd*1.5
gmu=rmu+sd*3
bmu=maxe-sd*1.5
uvmu=maxe+sd*1.5