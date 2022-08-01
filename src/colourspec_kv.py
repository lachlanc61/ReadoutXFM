from email.quoprimime import body_check
from tkinter import X
import numpy as np
import math
import os
import itertools as it
import csv
from scipy.stats import norm
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
#workdir
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script

#figure params
colourmap='viridis'    #colourmap for figure
#colourmap='copper'    #colourmap for figure
figx=20         #cm width of figure
figy=10         #cm height of figure
smallfont = 8
medfont = 10
lgfont = 12
lwidth = 1  #default linewidth
bwidth = 1  #default border width


colstart=1.041  #Na     
colend=17.441   #Mo
#-------------------------------------
#FUNCTIONS
#-----------------------------------


def ngauss(x, mu, sig1, amp):
    g1=norm.pdf(x, mu, sig1)
    g1n=np.divide(g1,max(g1))
    return np.multiply(g1n, amp)

#    return np.piecewise(x, [x < mu, x >= mu], [gauss(x, mu, sig1), gauss(x, mu, sig2)])

#-----------------------------------
#MAIN START
#-----------------------------------

#initialise directories relative to script
script = os.path.realpath(__file__) #_file = current script
spath=os.path.dirname(script) 
wdir=os.path.join(spath,wdirname)
odir=os.path.join(spath,odirname)
print("script:", script)
print("script path:", spath)
print("data path:", wdir)


#plot defaults
plt.rc('font', size=smallfont)          # controls default text sizes
plt.rc('axes', titlesize=smallfont)     # fontsize of the axes title
plt.rc('axes', labelsize=medfont)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('legend', fontsize=smallfont)    # legend fontsize
plt.rc('figure', titlesize=lgfont)  # fontsize of the figure title
plt.rc('lines', linewidth=lwidth)
plt.rcParams['axes.linewidth'] = bwidth


fig=plt.figure()
ax=fig.add_subplot(111)

#ax.set_yscale('log')
ax.set_ylabel('intensity (counts)')
ax.set_xlim(-5,25)
ax.set_xlabel('energy (keV)')
#ax.set_xscale('log')


ff="out_30000.dat"
f = os.path.join(wdir, ff)
fname = os.path.splitext(ff)[0]
print(fname)

#x entending variables (for ir gaussian)
minx=-5
stepx=0.01
xzer=np.floor(-(minx/stepx)).astype(int)


mine=1.04
maxe=17.44
sds=9
sd=(maxe-mine)/(sds)

irmu=mine-sd*1.5
rmu=mine+sd*1.5
gmu=rmu+sd*3
bmu=maxe-sd*1.5
uvmu=maxe+sd*1.5
print("MUs")
print(irmu,rmu,gmu,bmu,uvmu)

#width=(bmu-rmu)/(2*3)

if (os.path.isfile(f)) and (f.endswith(".dat")):
    print(f)
    spectrum = np.genfromtxt(f, dtype="float64")
    e=spectrum[:,0]
    y=spectrum[:,1]
    

    #max of ir curve is outside e
    #need to extend to -5 to normalise correctly
    xe=np.arange(minx,0,stepx)
    xe=np.append(xe,e)

    #create it, then truncate back
    ir=ngauss(xe, irmu, sd, max(y))
    ir=ir[xzer:]
    
    red=ngauss(e, rmu, sd, max(y))
    green=ngauss(e, gmu, sd, max(y))
    blue=ngauss(e, bmu, sd, max(y))
    uv=ngauss(e, uvmu, sd, max(y))
    sum=np.add(red,green)
    sum=np.add(sum, blue)
    sum=np.add(sum, ir)
    sum=np.add(sum, uv)

    rch=np.zeros(len(e))
    gch=np.zeros(len(e))
    bch=np.zeros(len(e))

    rs=np.zeros(len(e))
    gs=np.zeros(len(e))
    bs=np.zeros(len(e))
    
    #initialise yield model (yield=1, bg=0 leaves unchanged)
    yld=np.ones(len(e))
    bg=np.zeros(len(e))
    #NB yield model would go here
    #NB background model would go here

    #calc background- and yield- adjusted spectrum
    yc=np.subtract(y,bg)
    yc=np.multiply(yc,yld)

    #calculate RGB matrices
    for i in np.arange(len(e)):
        rch[i]=yc[i]*(red[i]+uv[i])
        gch[i]=yc[i]*(green[i])
        bch[i]=yc[i]*(blue[i]+ir[i])
        rs[i]=rs[(i-1)]+rch[i]
        gs[i]=gs[(i-1)]+gch[i]
        bs[i]=bs[(i-1)]+bch[i]

    rval=np.sum(rch)/len(e)
    gval=np.sum(gch)/len(e)
    bval=np.sum(bch)/len(e)
    mval=max([rval,gval,bval])
    rval=rval/mval
    gval=gval/mval
    bval=bval/mval

    print(rval,gval,bval)    

    ms=max([max(rs),max(gs),max(bs)])

    rs=np.divide(rs,ms/max(y))
    gs=np.divide(gs,ms/max(y))
    bs=np.divide(bs,ms/max(y))

ax.plot(e, y, color = (rval,gval,bval), label="data")
ax.plot(e, rs, color="#FF0000")
ax.plot(e, gs, color="#00FF00")
ax.plot(e, bs, color="#0000FF")
ax.plot(e, red, '--', color="#FF0000")
ax.plot(e, green, '--', color="#00FF00")
ax.plot(e, blue, '--', color="#0000FF")
ax.plot(e, ir, '--', color="#0000FF")
ax.plot(e, uv, '--', color="#FF0000")
#ax.plot(e, sum, '--', color="orange")

plt.show()

exit()

"""""
next step:
scale all colours relative to max (r+g+b)? across all f

then map with those

long term:

scale i with yields
fit background using SNIP?
https://stackoverflow.com/questions/57350711/baseline-correction-for-spectroscopic-data
deal w/ Mo+compton somehow
scale gaussians w/ k spacings

"""""