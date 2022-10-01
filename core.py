import os
import time
import gc
import time

import numpy as np

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

"""

#-----------------------------------
#CLASSES
#-----------------------------------

#-----------------------------------
#INITIALISE
#-----------------------------------

#initialise directories relative to current script
f, fname, script, spath, wdir, odir = utils.initialise()

starttime = time.time()             #init timer
chan=np.arange(0,config.NCHAN)      #channels
energy=chan*config.ESTEP            #energy list
noisecorrect=True                   #apply adjustment to SNIP to fit noisy pixels

#if we are creating colourmaps, set up colour routine
if config.DOCOLOURS == True: colour.initialise(energy)


#-----------------------------------
#MAIN START
#-----------------------------------

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
            = bitops.parsespec(stream, headerlen, chan, energy, mapx, mapy, totalpx, odir)
    else:   
        #read these from a text file
        data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows \
            = bitops.readspec(config.outfile, odir)

    #show memory usage    
    utils.varsizes(locals().items())

    #manually drop the bytestream from memory
    #   clustering is memory intensive, better to get this removed asap
    del stream
    gc.collect()

    #perform post-analysis:

    #create and show colour map
    if config.DOCOLOURS == True:
        rgbarray=colour.complete(rvals, gvals, bvals, mapx, nrows, odir)

    #perform clustering
    if config.DOCLUST:
        categories, classavg = clustering.complete(data, energy, totalpx, mapx, mapy, odir)

print("CLEAN EXIT")
exit()


"""
runtime log:
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