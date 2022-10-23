import os
import time
import gc
import time

import numpy as np

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
#vars
#-----------------------------------
CONFIG_FILE='config.yaml'

#-----------------------------------
#INITIALISE
#-----------------------------------

config=utils.readcfg(CONFIG_FILE)

#initialise read file and all directories relative to current script
script, spath, wdir, odir = utils.initdirs(config)

fi, fname, fo, oname = utils.initfiles(config, wdir, odir)

starttime = time.time()             #init timer

noisecorrect=True                   #apply adjustment to SNIP to fit noisy pixels

#-----------------------------------
#MAIN START
#-----------------------------------


#initialise map
#   parses header into map.headerdict
#       puts pointer (map.idx) at start of first pixel record
map = bitops.Map(config, fi, fo)

#initialise the spectrum-by-pixel container
#       WARNING: large memory spike here if map is big
#       pre-creates all arrays for storing data, pixel header values etc
pixelseries = bitops.PixelSeries(config, map)

#if we are creating colourmaps, set up colour routine
if config['DOCOLOURS'] == True: colour.initialise(config, map.energy)

#   if we are skipping some of the file
#       assign the ratio and adjust totalpx
if config['SHORTRUN']:
    skipratio=config['shortpct']/100
    trunc_y=int(np.ceil(map.yres*skipratio))
    map.numpx=map.xres*trunc_y
    print(f"SHORT RUN: ending at {skipratio*100} %")


print(f"pixels expected: {map.numpx}")
print("---------------------------")

#BEGIN PARSING
starttime = time.time() 

#if we are parsing the .GeoPIXE file
if config['FORCEPARSE']:
    try:
        map.parse(config, pixelseries)
    finally:
        map.closefiles()

else:   
    map.read(config, odir)

runtime = time.time() - starttime

pixelseries.exportheader(config, odir)

if config['SAVEPXSPEC']:
    pixelseries.exportseries(config, odir)

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

#show memory usage    
utils.varsizes(locals().items())

exit()

#manually drop the bytestream from memory
#   clustering is memory intensive, better to get this removed asap
del stream
gc.collect()

#perform post-analysis:

#create and show colour map
if config['DOCOLOURS'] == True:
    rgbarray=colour.complete(rvals, gvals, bvals, mapx, nrows, odir)

#perform clustering
if config['DOCLUST']:
    categories, classavg = clustering.complete(config, data, energy, totalpx, mapx, mapy, odir)

infile.close()
outfile.close()

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
    snip:                   0.002734 s
    complex snip:           0.002919 s

OO:
    map+pxseries:           0.001852 s


"""