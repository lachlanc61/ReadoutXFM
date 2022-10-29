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
- projects spectra onto simple RGB channels
- displays as RGB

./data has example dataset

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
pixelseries = parser.PixelSeries(config, map)

#if we are creating colourmaps, set up colour routine
if config['DOCOLOURS'] == True: colour.initialise(config, map.energy)

#   if we are skipping some of the file
#       assign the ratio and adjust totalpx
if config['SHORTRUN']:
    skipratio=config['shortpct']/100
    trunc_y=int(np.ceil(map.yres*skipratio))
    map.numpx=map.xres*trunc_y
    print(f"SHORT RUN: ending at {skipratio*100} %")

#BEGIN PARSING

#start a timer
starttime = time.time() 

#if we are parsing the .GeoPIXE file
#   begin parsing
if config['FORCEPARSE']:
    try:
        map.parse(config, pixelseries)
    finally:
        map.closefiles()
#else we are reading from a pre-parsed csv
#   do that instead
else:   
    map.read(config, odir)

runtime = time.time() - starttime

print(
"---------------------------\n"
"MAP COMPLETE\n"
"---------------------------\n"
f"pixels expected (X*Y): {map.numpx}\n"
f"pixels found: {pixelseries.npx}\n"
f"total time: {round(runtime,2)} s\n"
f"time per pixel: {round((runtime/pixelseries.npx),6)} s\n"
"---------------------------"
)

pixelseries.exportheader(config, odir)

if config['SUBMAPONLY']:
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
    rgbarray=colour.complete(pixelseries.rvals, pixelseries.gvals, pixelseries.bvals, map.xres, pixelseries.nrows, odir)

#perform clustering
if config['DOCLUST']:
    categories, classavg = clustering.complete(config, pixelseries.data, map.energy, map.numpx, map.xres, map.yres, odir)

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

"""