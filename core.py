import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import struct 
import time
from decimal import *
from scipy.optimize import curve_fit
from src.utils import *

import seaborn as sns
import time

from sklearn import datasets, decomposition, manifold, preprocessing
from colorsys import hsv_to_rgb

import umap.umap_ as umap
"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- projects spectra onto simple RGB channels
- displays as RGB

./data has example dataset

SPEED
                nrec    ttot    t/rec
reading only:   8700    14      0.00014 
colourmap:      8700    67      0.0078

"""
#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
#global variables
FTYPE=".GeoPIXE"    #valid: ".GeoPIXE"
DEBUG=False     #debug flag (pixels)
DEBUG2=False    #second-level debug flag (channels)
PXHEADERLEN=16  #pixel header size
PXFLAG="DP"
NCHAN=4096
ESTEP=0.01
CHARENCODE = 'utf-8'
DOCOLOURS=False
DOCLUST=True

#MAPX=128   #for leaf
#MAPY=68
MAPX=256    #for geo2
MAPY=126

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
#infile = "leaf2_overview.GeoPIXE"    #assign input file
infile = "geo2.GeoPIXE"    #assign input file

detid="A"   #detector ID - not needed for single detector maps

#colour-related variables
mine=1.04   #minimum energy of interest
minxe=-5    #extended minimum x for ir
elastic=17.44  #energy of tube Ka
maxe=30  #maximum energy of interest
sds=9   #standard deviations
rgblogscale=True    #map RGB as log of intensity

#figure params
colourmap='Set1'    #colourmap for figure
#colourmap='copper'    #colourmap for figure
figx=20         #cm width of figure
figy=10         #cm height of figure
smallfont = 8   #default small font
medfont = 10    #default medium font
lgfont = 12     #default large font
lwidth = 1  #default linewidth
bwidth = 1  #default border width

#debug - skip all but total/skipratio
skipratio=1

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def binunpack(stream, idx, sformat):
    """
    parse binary data via struct.unpack
    takes:
        stream of bytes
        byte index
        format flag for unpack (currently accepts: <H <f <I )
    returns:
        value in desired format (eg. int, float)
        next byte index
    """

    if sformat == "<H":
        nbytes=2
    elif sformat == "<f":
        nbytes=4
    elif sformat == "<I":
        nbytes=4
    else:
        print(f"ERROR: {sformat} not recognised by local function binunpack")
        exit(0)

    #struct unpack outputs tuple
    #want int so take first value
    retval = struct.unpack(sformat, stream[idx:idx+nbytes])[0]
    idx=idx+nbytes
    return(retval, idx)    

def readpxrecord(idx, stream):
    """"
    Pixel Record
    Note: not name/value pairs for file size reasons. The pixel record header is the only record type name/value pair, for easier processing. We are keeping separate records for separate detectors, since the deadtime information will also be per detector per pixel.
        1.	Record type pair  "DP", Length of pixel data record in bytes ( 4 byte int)
        2.	X                          Horizontal pixel index (2 byte int)
        3.	Y                          Vertical pixel index (2 byte int)
        4.	Detector               Data in this record is for this detector (2 byte int)
        5.	Deadtime             Deadtime for this pixel (4 byte float)
        6.	Data (for each channel with data up to maximum channel index)
            a.	Channel     Channel index (0- Max Chan) (2 byte int)
            b.	Count         Event counts in channel (2 byte int)

    #   concise format:
    #   DP  len     X       Y       det     dt  DATA
    #   2c  4i     2i       2i      2i      4f
    """
    """
    Read binary with struct
    https://stackoverflow.com/questions/8710456/reading-a-binary-file-with-python
    Read binary as chunks
    https://stackoverflow.com/questions/71978290/python-how-to-read-binary-file-by-chunks-and-specify-the-beginning-offset
    """
    pxstart=idx
#   check for pixel start flag "DP" at first position after header:
    #   unpack first two bytes after header as char
    pxflag=struct.unpack("cc", stream[idx:idx+2])[:]
    #   use join to merge into string
    pxflag="".join([pxflag[0].decode(CHARENCODE),pxflag[1].decode(CHARENCODE)])

    #   check if string is "DP" - if not, fail
    if pxflag != PXFLAG:
        print(f"ERROR: pixel flag 'DP' expected but not found at byte {idx}")
        exit()
    else:
        if (DEBUG): print(f"pixel at: {idx} bytes")

    idx=idx+2   #step over "DP"
    if (DEBUG): print(f"next bytes at {idx}: {stream[idx:idx+PXHEADERLEN]}")
    #read each header field and step idx to end of field
    pxlen, idx=binunpack(stream,idx,"<I")
    xcoord, idx=binunpack(stream,idx,"<H")
    ycoord, idx=binunpack(stream,idx,"<H")
    det, idx=binunpack(stream,idx,"<H")
    dt, idx=binunpack(stream,idx,"<f")

    #print header fields
    if (DEBUG): 
        print("PXLEN: ",pxlen)
        print("XCOORD: ",xcoord)
        print("YCOORD: ",ycoord)
        print("DET: ",det)
        print("DT: ",dt)

    #initialise channel index and result arrays
    j=0
    chan=np.zeros(int((pxlen-PXHEADERLEN)/4), dtype=int)
    counts=np.zeros(int((pxlen-PXHEADERLEN)/4), dtype=int)
    #       4 = no. bytes in each x,y pair
    #         = 2x2 bytes each 

    #iterate until byte index passes pxlen
    #pull channel, count pairs
    while idx < (pxstart+pxlen):
   #while idx < 2000:
        if (DEBUG2): print(f"next bytes at {idx}: {stream[idx:idx+8]}")
        chan[j], idx=binunpack(stream,idx,"<H")
        counts[j], idx=binunpack(stream,idx,"<H")
        if (DEBUG2): print(f"idx {idx} x {chan[j]} y {counts[j]}")
        
    #    if (DEBUG): print(f"idx {idx} / {pxstart+pxlen}")
        j=j+1
    if (DEBUG): print(f"following bytes at {idx}: {stream[idx:idx+10]}")
    return(chan, counts, pxlen, xcoord, ycoord, det, dt, idx)



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
    xe=np.arange(-5,0,ESTEP)
    xe=np.append(xe,e)
    
    #create ir gaussian, then truncate back
    ir=normgauss(xe, irmu, sd, max(y))
    ir=ir[xzer:]

    #create other gaussians
    #   currently depends on a lot of variables outside function
    #   should spin this off into own script and put variable definitions there
    red=normgauss(e, rmu, sd, max(y))
    green=normgauss(e, gmu, sd, max(y))
    blue=normgauss(e, bmu, sd, max(y))
    uv=normgauss(e, uvmu, sd, max(y))

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






#-----------------------------------
#CLASSES
#-----------------------------------
"""
reducers = [
    (manifold.TSNE, {"perplexity": 50}),
    # (manifold.LocallyLinearEmbedding, {'n_neighbors':10, 'method':'hessian'}),
    (manifold.Isomap, {"n_neighbors": 30}),
    (manifold.MDS, {}),
    (decomposition.PCA, {}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
]
"""
reducers = [
    (decomposition.PCA, {}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
]


#-----------------------------------
#INITIALISE
#-----------------------------------


#initialise directories relative to script
script = os.path.realpath(__file__) #_file = current script
spath=os.path.dirname(script) 
wdir=os.path.join(spath,wdirname)
odir=os.path.join(spath,odirname)
print("script:", script)
print("script path:", spath)
print("data path:", wdir)
print("---------------")


#plot defaults

if False:
    plt.rc('font', size=smallfont)          # controls default text sizes
    plt.rc('axes', titlesize=smallfont)     # fontsize of the axes title
    plt.rc('axes', labelsize=medfont)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
    plt.rc('legend', fontsize=smallfont)    # legend fontsize
    plt.rc('figure', titlesize=lgfont)  # fontsize of the figure title
    plt.rc('lines', linewidth=lwidth)
    plt.rcParams['axes.linewidth'] = bwidth

"""
plot options for spectra - not currently usede
fig=plt.figure()
ax=fig.add_subplot(111)
"""
"""
ax.set_yscale('log')

ax.set_ylabel('intensity (counts)')
#ax.set_ylim(0,50)

#ax.set_xscale('log')
ax.set_xlim(0,40)
ax.set_xlabel('energy (keV)')
"""

starttime = time.time() #initialise timer
totalpx=MAPX*MAPY   # map size

chan=np.arange(0,NCHAN)
energy=chan*ESTEP

#initialise RGB split variables
if DOCOLOURS == True:
    ncols=5
    xzer=np.floor(-(minxe/ESTEP)).astype(int)
    sd=(maxe-mine)/(sds)
    irmu=mine-sd*1.5
    rmu=mine+sd*1.5
    gmu=rmu+sd*3
    bmu=maxe-sd*1.5
    uvmu=maxe+sd*1.5


#-----------------------------------
#MAIN START
#-----------------------------------

#check filetype is recognised - currently only accepts .GeoPIXE
if FTYPE == ".GeoPIXE":
    f = os.path.join(wdir,infile)
    fname = os.path.splitext(os.path.basename(f))[0]

    print("opening .geo:",fname)
else: 
    print(f'FATAL: filetype {FTYPE} not recognised')
    exit()

print("---------------")

with open(f, mode='rb') as file: # rb = read binary
    
    stream = file.read()    #NB. to read in chunks, add chunk size as read(SIZE)
    streamlen=len(stream)
    print("stream length in bytes ",streamlen)
    print("first two bytes: ",stream[:2])

    headerlen=binunpack(stream,0,"<H")[0]
    print(f"header length: {headerlen}")
    
    #occasionally, files might be missing the header
    #   when this happens, the first bytes are "DP" - denoting the start of a pixel record
    #   therefore if we get 20550 (="DP" as <uint16), header is missing
    if headerlen == 20550:
        print("WARNING: no header found")
        headerlen=0
    
    #assign starting index 
    idx=headerlen+2 #legnth of header + 2 bytes

    #initialise pixel param arrays
    pxlen=np.zeros(totalpx)
    xidx=np.zeros(totalpx)
    yidx=np.zeros(totalpx)
    det=np.zeros(totalpx)
    dt=np.zeros(totalpx)
    
    if DOCOLOURS == True:
        #initalise pixel colour arrays
        rvals=np.zeros(totalpx)
        gvals=np.zeros(totalpx)
        bvals=np.zeros(totalpx)
        totalcounts=np.zeros(totalpx)

    data=np.zeros((totalpx,NCHAN))

    i=0 #pixel counter

    #loop through pixels
    while idx < streamlen:
        if i % 50 == 0: print(f"Pixel {i} at {idx} bytes")
        #read pixel record
        #   output: spectrum, all header params, finishing index
        chan, counts, pxlen[i], xidx[i], yidx[i], det[i], dt[i], idx = readpxrecord(idx, stream)

        #fill gaps in spectrum 
        #   (ie. all chans where y=0 are missing, add them back)
        chan, counts = gapfill(chan,counts, NCHAN)

        #convert chan to energy
        #      easier to do this after gapfill so dict doesn't have to deal with floats
        data[i,:]=counts

        #build colours if required
        if DOCOLOURS == True: rvals[i], bvals[i], gvals[i], totalcounts[i] = spectorgb(energy, counts)
        
        #warn if i is unexpectedly high - would mostly happen if header is wrong
        if i > totalpx:
            print(f"WARNING: pixel count {i} exceeds expected map size {totalpx}")
        #pixel outputs:
        #ax.plot(energy, data[i,:], color = "red", label=i)
#        print("index at end of record",idx)

        #if idx > 500:
        if idx > streamlen/skipratio:
            print("ending at:", idx)
            idx=streamlen+1
        i+=1

    print("---------------------------")
    print("MAP COMPLETE")
    print("---------------------------")
    #output result arrays   
    runtime = time.time() - starttime
    print("pixels expected (X*Y):", totalpx) 
    print("pixels found:", i)
    print(f"total time: {round(runtime,2)} s")
    print(f"time per pixel: {round((runtime/i),6)} s") 
    print("---------------------------")
    print("pixel lengths")
    print(pxlen[:i])
    print("xidx")
    print(xidx[:i])
    print("yidx")
    print(yidx[:i])
    print("detector")
    print(det[:i])
    print("dt")
    print(dt[:i])    

    if DOCOLOURS == True:
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

        np.savetxt(os.path.join(odir, "rvals.txt"), rvals)
        np.savetxt(os.path.join(odir, "gvals.txt"), gvals)
        np.savetxt(os.path.join(odir, "bvals.txt"), bvals)

        rreshape=np.reshape(rvals, (-1, MAPX))
        greshape=np.reshape(gvals, (-1, MAPX))
        breshape=np.reshape(bvals, (-1, MAPX))

        rgbarray = np.zeros((MAPY,MAPX,3), 'uint8')
        rgbarray[..., 0] = rreshape*256
        rgbarray[..., 1] = greshape*256
        rgbarray[..., 2] = breshape*256
        
        plt.imshow(rgbarray)

        plt.show()


    if DOCLUST:
        
        sns.set(context="paper", style="white")
        n_cols = len(reducers)
        counter = 0
        ax_list = []
        elements=np.arange(0,MAPX*MAPY)
        # plt.figure(figsize=(9 * 2 + 3, 12.5))
        plt.figure(figsize=(10, 8))
        plt.subplots_adjust(
            left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
        )
        #for data, labels in test_data:
        #    print("cycle start",ax_index)
        for reducer, args in reducers:

            start_time = time.time()
            embedding = reducer(n_components=2, **args).fit_transform(data)
            elapsed_time = time.time() - start_time
            ax = plt.subplot(1, n_cols, (counter+1))
            print(embedding)

            ax.scatter(*embedding.T, s=10, c=elements, cmap="Spectral", alpha=0.5)
            #else:
            #    ax.scatter(*embedding.T, s=10, c="red", cmap="Spectral", alpha=0.5)
            ax.text(
                0.99,
                0.01,
                "{:.2f} s".format(elapsed_time),
                transform=ax.transAxes,
                size=14,
                horizontalalignment="right",
            )
            
            redname=repr(reducers[counter][0]()).split("(")[0]
            print("reducer",redname, reducer)
            ax_list.append(ax)
            ax_list[counter].set_xlabel(redname, size=16)
            ax_list[counter].xaxis.set_label_position("top")
            print("rname:",redname)
            np.savetxt(os.path.join(odir, redname + ".txt"), embedding)
            
            counter += 1
           
        plt.setp(ax_list, xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()


    np.savetxt(os.path.join(odir, "pxlen.txt"), pxlen)
    np.savetxt(os.path.join(odir, "xidx.txt"), xidx)
    np.savetxt(os.path.join(odir, "yidx.txt"), yidx)
    np.savetxt(os.path.join(odir, "detector.txt"), det)
    np.savetxt(os.path.join(odir, "dt.txt"), dt)

print("CLEAN EXIT")
exit()
