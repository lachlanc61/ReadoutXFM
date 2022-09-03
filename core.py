import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import struct 
import time
import gc
from decimal import *
from scipy.optimize import curve_fit



import seaborn as sns
import time

from sklearn import datasets, decomposition, manifold, preprocessing
from colorsys import hsv_to_rgb

import umap.umap_ as umap


import config
import src.utils as utils
import src.bitops as bitops
import src.colour as colour
import src.clustering as clustering


"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- projects spectra onto simple RGB channels
- displays as RGB

./data has example dataset

SPEED
                t/px
reading only:   0.00014 
colourmap:      0.0078
read and clust  0.001296 

"""

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

starttime = time.time()             #init timer

totalpx=config.MAPX*config.MAPY     #map size
chan=np.arange(0,config.NCHAN)      #channels
energy=chan*config.ESTEP            #energy list

#-----------------------------------
#MAIN START
#-----------------------------------

#check filetype is recognised - currently only accepts .GeoPIXE
if config.FTYPE == ".GeoPIXE":
    f = os.path.join(config.wdir,config.infile)
    fname = os.path.splitext(os.path.basename(f))[0]

    print("opening .geo:",fname)
else: 
    print(f'FATAL: filetype {config.FTYPE} not recognised')
    exit()

print("---------------")


#open the datafile 
with open(f, mode='rb') as file: # rb = read binary
    
    #generate bytestream
    stream = file.read()         #NB. to read in chunks, add chunk size as read(SIZE)
    streamlen=len(stream)

    print("stream length in bytes ",streamlen)
    print("first two bytes: ",stream[:2])

    headerlen=bitops.binunpack(stream,0,"<H")[0]
    print(f"header length: {headerlen}")
    
    #check for missing header
    #   pixels start with "DP" (=20550 as <uint16)
    #   if we find this immediately, header is zero length
    if headerlen == 20550:
        print("WARNING: no header found")
        headerlen=0
    
    #assign starting pixel index 
    idx=headerlen+2 #legnth of header + 2 bytes

    #initialise pixel param arrays
    pxlen=np.zeros(totalpx)
    xidx=np.zeros(totalpx)
    yidx=np.zeros(totalpx)
    det=np.zeros(totalpx)
    dt=np.zeros(totalpx)
    
    if config.DOCOLOURS == True:
        #initalise pixel colour arrays
        rvals=np.zeros(totalpx)
        gvals=np.zeros(totalpx)
        bvals=np.zeros(totalpx)
        totalcounts=np.zeros(totalpx)

    #initialise data array
    data=np.zeros((totalpx,config.NCHAN))

    i=0 #pixel counter

    #loop through pixels
    while idx < streamlen:

        #print pixel index every 50 px
        if i % 50 == 0: print(f"Pixel {i} at {idx} bytes")

        #read pixel record into spectrum and header param arrays, 
        # + reassign index at end of read
        outchan, counts, pxlen[i], xidx[i], yidx[i], det[i], dt[i], idx = bitops.readpxrecord(idx, stream)

        #fill gaps in spectrum 
        #   (ie. add 0s for all missing chans)
        outchan, counts = utils.gapfill(outchan,counts, config.NCHAN)

        #warn if recieved channel list is different length to chan array
        if len(outchan) != len(chan):
            print("WARNING: channel list from pixel does not match expected")
      
        #assign counts into data array - 
        data[i,:]=counts

        #build colours if required
        if config.DOCOLOURS == True: rvals[i], bvals[i], gvals[i], totalcounts[i] = colour.spectorgb(energy, counts)
        
        #warn if i is unexpectedly high - would mostly happen if header is wrong
        if i > totalpx:
            print(f"WARNING: pixel count {i} exceeds expected map size {totalpx}")

        if (config.SHORTRUN == True) and (idx > streamlen*(config.skipratio/100)):
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

    #clear the bytestream from memory
    del stream
    gc.collect()

    if config.DOCOLOURS == True:
        colour.clcomplete(rvals, gvals, bvals, totalcounts)

    print("DOCLUST", config.DOCLUST)
    if config.DOCLUST:
        embedding, clusttimes = clustering.reduce(data)
        categories = clustering.dokmeans(embedding)
        print("categories full")
        print(categories)
        clustering.clustplt(embedding, categories, clusttimes)


    np.savetxt(os.path.join(config.odir, "pxlen.txt"), pxlen)
    np.savetxt(os.path.join(config.odir, "xidx.txt"), xidx)
    np.savetxt(os.path.join(config.odir, "yidx.txt"), yidx)
    np.savetxt(os.path.join(config.odir, "detector.txt"), det)
    np.savetxt(os.path.join(config.odir, "dt.txt"), dt)

print("CLEAN EXIT")
exit()
