from tkinter import E
import numpy as np
import os
import itertools as it
import csv
from scipy.stats import norm

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

minx=-5
stepx=0.01

mine=1.04
maxe=17.44
sds=9

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def ngauss(x, mu, sig1, amp):
    g1=norm.pdf(x, mu, sig1)
    g1n=np.divide(g1,max(g1))
    return np.multiply(g1n, amp)


#-------------------------------------
#DERIVED VARIABLES
#-----------------------------------
#ncols=len(steps)+2
ncols=5

xzer=np.floor(-(minx/stepx)).astype(int)
sd=(maxe-mine)/(sds)

irmu=mine-sd*1.5
rmu=mine+sd*1.5
gmu=rmu+sd*3
bmu=maxe-sd*1.5
uvmu=maxe+sd*1.5

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
ax.set_yscale('log')

ax.set_ylabel('intensity (counts)')
ax.set_xlim(0,25)

#ax.set_xscale('log')
ax.set_xlim(1.5,25)
ax.set_xlabel('energy (keV)')


#istart=0
#istep=1
#imax=10

istart=1
istep=5000
imax=64503

npx=(imax-istart)/istep
#istep=5000
#imax=64503

ilines=np.floor((imax-istart)/istep).astype('int')+1

rvals=np.arange(npx)
gvals=np.arange(npx)
bvals=np.arange(npx)
ysum=np.arange(npx)

ff="temp.csv"

#split filenames / paths
f = os.path.join(wdir, ff)
fname = os.path.splitext(ff)[0]
print(fname)

#END if file doesn't exist or is not csv
if (not os.path.isfile(f)) or (not f.endswith(".csv")): 
    print("FATAL: file {} not found/not csv".format(f))
    exit()

oneline=np.genfromtxt(f, delimiter=',',max_rows=2)
spectra=np.zeros((ilines,len(oneline[1,:])))

e=np.arange(len(spectra[1,1:]))
e=e*0.01

#open the file as csv
with open(f, encoding='UTF-8') as f_in:
    r=csv.reader(f_in)
    i=0
    
    #ITERATE THROUGH PIXELS
    for row in it.islice(r,istart,None,istep):
        print("index",i)
        spectra[i] = np.genfromtxt(row, dtype=float, delimiter=',')
        
        # e=spectra[:,0]
        y=spectra[i,1:]

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

        #initialise channel outputs
        rch=np.zeros(len(e))
        gch=np.zeros(len(e))
        bch=np.zeros(len(e))

        #initialise sums for plotting
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
        
        for j in np.arange(len(e)):
            rch[j]=yc[j]*(red[j]+uv[j])
            gch[j]=yc[j]*(green[j])
            bch[j]=yc[j]*(blue[j]+ir[j])
        #    rs[j]=rs[(j-1)]+rch[j]
        #    gs[j]=gs[(j-1)]+gch[j]
        #    bs[j]=bs[(j-1)]+bch[j]
        
        #calculate average per channel
        rvals[i]=np.sum(rch)/len(e)
        gvals[i]=np.sum(gch)/len(e)
        bvals[i]=np.sum(bch)/len(e)
        ysum[i]=np.sum(yc)
        
        #get max channel and normalise to this
        #mval=max([rvals[i],gvals[i],bvals[i]])
        #rval=rval/mval
        #gval=gval/mval
        #bval=bval/mval

        #final output colours


        #normalise sums for display
        #ms=max([max(rs),max(gs),max(bs)])
        #rs=np.divide(rs,ms/max(y))
        #gs=np.divide(gs,ms/max(y))
        #bs=np.divide(bs,ms/max(y))

        #index/initialise variables
        i+=1
        y=np.zeros(len(y))
            
        #print(spectra)       

#print(rvals,gvals,bvals,ysum)
allch=np.append(rvals,gvals)   
allch=np.append(allch,bvals)  
chmax=max(allch)
#gmax=max(gvals)
#bmax=max(bvals)
mysum=max(ysum)

print(np.arange(len(ysum)))

for i in np.arange(len(ysum)):
    print("index",i)
#colorVal = scalarMap.to_rgba(i)
    idx=spectra[i,0].astype('int')
    mult=ysum[i]/mysum
    y=spectra[i,1:]
    ax.plot(e, y, color = (rvals[i]*mult/chmax,gvals[i]*mult/chmax,bvals[i]*mult/chmax), label=idx)
    print("RGB",rvals[i]*mult/chmax,gvals[i]*mult/chmax,bvals[i]*mult/chmax)
plt.show()