import os
import time
import gc
import time
import json

import numpy as np
from sklearn import decomposition
import umap.umap_ as umap

import config
import src.utils as utils
import src.bitops as bitops
import src.colour as colour
import src.clustering as clustering
import src.fitting as fitting
"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- projects spectra onto simple RGB channels
- displays as RGB

./data has example dataset

SPEEDUP
                            t/px
reading only:               0.000140 s
+clustering                 0.001296 s     
colourmap:                  0.007800 s

improving colourmap:    
    for j:                  0.007625 s
    vectorise channels:     0.004051 s
    pre-init gaussians:     0.002641 s   
    fully vectorised:       0.001886 s
add background fitting:
    snip:               0.002734 s
    complex snip:       0.002919 s

"""

#-----------------------------------
#CLASSES
#-----------------------------------

#-----------------------------------
#INITIALISE
#-----------------------------------

starttime = time.time()             #init timer
chan=np.arange(0,config.NCHAN)      #channels
energy=chan*config.ESTEP            #energy list
noisecorrect=True                   #apply adjustment to SNIP to fit noisy pixels

#if we are creating colourmaps, set up colour routine
if config.DOCOLOURS == True: colour.initialise(energy)

#-----------------------------------
#MAIN START
#-----------------------------------

#check filetype is recognised - currently only accepts .GeoPIXE
if config.FTYPE == ".GeoPIXE":
    f = os.path.join(config.wdir,config.infile)
    fname = os.path.splitext(os.path.basename(f))[0]
    print(f"Opening file: {fname}\n")
else: 
    print(f'FATAL: filetype {config.FTYPE} not recognised')
    exit()

#open the datafile 
with open(f, mode='rb') as file: # rb = read binary

    #generate bytestream
    stream = file.read()         #NB. to read in chunks, add chunk size as read(SIZE)
    streamlen=len(stream)

    headerlen, mapx, mapy, totalpx = bitops.readgpxheader(stream)

    #   if we are skipping some of the file
    #       assign the ratio and adjust totalpx
    if config.SHORTRUN:
        skipratio=config.shortpct/100
        trunc_y=int(np.ceil(mapy*skipratio))
        totalpx=mapx*trunc_y
        print(f"SHORT RUN: ending at {skipratio*100} %")

    print(f"pixels expected: {totalpx}")
    print("---------------------------")

    #if we are parsing the .GeoPIXE file
    if config.FORCEPARSE:

        #loop through all pixels and return data, corrected data
        #   and pixel header arrays
        data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows \
            = bitops.readspectra(stream, headerlen, chan, energy, mapx, mapy, totalpx)

    else:   #read these from a text file
        print("loading from file", config.savename)
        data = np.loadtxt(os.path.join(config.odir, config.savename), dtype=np.uint16)
        pxlen=np.loadtxt(os.path.join(config.odir, "pxlen.txt"), dtype=np.uint16)
        xidx=np.loadtxt(os.path.join(config.odir, "xidx.txt"), dtype=np.uint16)
        yidx=np.loadtxt(os.path.join(config.odir, "yidx.txt"), dtype=np.uint16)
        det=np.loadtxt(os.path.join(config.odir, "detector.txt"), dtype=np.uint16)
        dt=np.loadtxt(os.path.join(config.odir, "dt.txt"), dtype=np.uint16)
        print("loaded successfully", config.savename)

    #print memory usage at this point    
    "---------------------------\n"
    "Memory usage:\n"
    "---------------------------\n"
    utils.varsizes(locals().items())

    #clear the bytestream from memory
    del stream
    gc.collect()

    #perform post-analysis:

    #create and show colour map
    if config.DOCOLOURS == True:
        rgbarray=colour.complete(rvals, gvals, bvals, mapx, nrows)

    #perform clustering
    if config.DOCLUST:
        categories, classavg = clustering.complete(data, energy, totalpx, mapx, mapy)

print("CLEAN EXIT")
exit()
