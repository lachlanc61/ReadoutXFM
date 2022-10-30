import os
import time
import gc
import time
import argparse
import numpy as np

import src.utils as utils
import src.parser as parser
import src.colour as colour
import src.clustering as clustering
import src.obj as obj
"""
Parses spectrum-by-pixel maps from IXRF XFM

- parses binary .GeoPIXE files
- extracts pixel parameters
- extracts pixel data
- classifies data via PCA and UMAP
- displays classified maps
- produces average spectrum per class

./data has example datasets

"""

#-----------------------------------
#vars
#-----------------------------------
CONFIG_FILE='config.yaml'

#-----------------------------------
#INITIALISE
#-----------------------------------

config=utils.readcfg(CONFIG_FILE)
argsparsed = argparse.ArgumentParser()

config, args=utils.readargs(config, argsparsed)

#initialise read file and all directories relative to current script

config, fi, fname, fsub, odir = utils.initcfg(config, args)

starttime = time.time()             #init timer

#-----------------------------------
#MAIN START
#-----------------------------------

#initialise map object
#   parses header into map.headerdict
#   places pointer (map.idx) at start of first pixel record
xfmap = obj.Xfmap(config, fi, fsub)

#initialise the spectrum-by-pixel object
#       pre-creates all arrays for storing data, pixel header values etc
#       WARNING: big memory spike here if map is large
pixelseries = obj.PixelSeries(config, xfmap)

#BEGIN PARSING

#start a timer
starttime = time.time() 

#if we are parsing the .GeoPIXE file
#   begin parsing
if config['FORCEPARSE']:
    try:
        pixelseries = xfmap.parse(config, pixelseries)
    finally:
        xfmap.closefiles()
#else read from a pre-parsed csv
else:   
    xfmap.read(config, odir)

runtime = time.time() - starttime

print(
"---------------------------\n"
"MAP COMPLETE\n"
"---------------------------\n"
f"pixels expected (X*Y): {xfmap.numpx}\n"
f"pixels found: {pixelseries.npx}\n"
f"total time: {round(runtime,2)} s\n"
f"time per pixel: {round((runtime/pixelseries.npx),6)} s\n"
"---------------------------"
)

pixelseries.exportheader(config, odir)

if not config['PARSEMAP']:
    print("WRITE COMPLETE")
    print("---------------------------")
    exit()

if config['SAVEPXSPEC']:
    pixelseries.exportseries(config, odir)

#show memory usage    
utils.varsizes(locals().items())

#perform post-analysis:

#create and show colour map
if config['DOCOLOURS'] == True:
    rgbarray=colour.complete(pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, xfmap.xres, pixelseries.nrows, odir)

#perform clustering
if config['DOCLUST']:
    categories, classavg = clustering.complete(config, pixelseries.data, xfmap.energy, xfmap.numpx, xfmap.xres, xfmap.yres, odir)

print("Processing complete")
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

w/ background fitting:
    snip:                   0.002734 s
    complex snip:           0.002919 s

OO:
    map+pxseries:           0.001852 s
    chunk parsing:          0.002505 s

refactored:
    read+write:             0.000643 s
"""