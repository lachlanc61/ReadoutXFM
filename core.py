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

"""
#CLASSES
class Map:
    def __init__(self, config, headerdict):
        self.xres = headerdict['File Header']['Xres']  #map dimension x
        self.yres = headerdict['File Header']['Yres']  #map dimension y
        self.xdim = headerdict['File Header']['Yres']
        self.ydim = headerdict['File Header']['Yres']
        self.channels
        self.energy
        self.npx = self.xdim*self.mapy 

class Pixels:
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

#assign input file object for reading
infile = open(fi, mode='rb') # rb = read binary
outfile = open(fo, mode='wb')   #wb = write binary

#generate initial bytestream
stream = infile.read()         
#stream = infile.read(config['chunksize'])   
streamlen=len(stream)

idx, headerdict = bitops.readgpxheader(stream)

#get map dimensions from header
try:
    mapx=headerdict['File Header']['Xres']  #map dimension x
    mapy=headerdict['File Header']['Yres']  #map dimension y
    totalpx=mapx*mapy   
    chan=np.arange(0,headerdict['File Header']['Chan'])      #channels
    energy=chan*headerdict['File Header']['Gain (eV)']/1000  #energy list
except:
    raise ValueError("FATAL: failure reading values from header")

#if we are creating colourmaps, set up colour routine
if config['DOCOLOURS'] == True: colour.initialise(config, energy)

#   if we are skipping some of the file
#       assign the ratio and adjust totalpx
if config['SHORTRUN']:
    skipratio=config['shortpct']/100
    trunc_y=int(np.ceil(mapy*skipratio))
    totalpx=mapx*trunc_y
    print(f"SHORT RUN: ending at {skipratio*100} %")

print(f"pixels expected: {totalpx}")
print("---------------------------")

#if we are parsing the .GeoPIXE file
if config['FORCEPARSE']:
    #loop through all pixels and return data, corrected data
    #   and pixel header arrays
    data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows \
        = bitops.parsespec(config, stream, idx, headerdict, odir)
else:   
    #read these from a text file
    data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows \
        = bitops.readspec(config, odir)

#show memory usage    
utils.varsizes(locals().items())

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
    snip:               0.002734 s
    complex snip:       0.002919 s
"""