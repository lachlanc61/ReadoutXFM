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

#-----------------------------------
#INITIALISE
#-----------------------------------

starttime = time.time()             #init timer
chan=np.arange(0,config.NCHAN)      #channels
energy=chan*config.ESTEP            #energy list

#if we are creating colourmaps, set up colour routine
if config.DOCOLOURS == True: colour.initialise(energy)

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

print(
    "---------------------------\n"
    "EXTRACTING SPECTRA\n"
    "---------------------------"
)

#open the datafile 
with open(f, mode='rb') as file: # rb = read binary
    
    #generate bytestream
    stream = file.read()         #NB. to read in chunks, add chunk size as read(SIZE)
    streamlen=len(stream)

    print(f"filesize: {streamlen} (bytes)")

    headerlen=bitops.binunpack(stream,0,"<H")[0]

    #check for header
    #   pixels start with "DP" (=20550 as <uint16)
    #   if we find this immediately, header is zero length
    #provided header is present
    #   read params from header
    if headerlen == 20550:
        print("WARNING: no header found")
        headerlen=0
        mapx=config.MAPX
        mapy=config.MAPY
        print("WARNING: map dimensions not found")
        print(f"-------using defaults {mapx},{mapy}")
    else:
        """
        if header present, read as json
        https://stackoverflow.com/questions/40059654/python-convert-a-bytes-array-into-json-format
        """
        #pull slice of byte stream corresponding to header
        #   bytes[0-2]= headerlen
        #   headerlen doesn't include trailing '\n' '}', so +2
        headerstream=stream[2:headerlen+2]
        #read it as utf8
        headerstream = headerstream.decode('utf8')
        
        #load into dictionary via json builtin
        headerdict = json.loads(headerstream)

        #create a human-readable dump for debugging
        headerdump = json.dumps(headerdict, indent=4, sort_keys=False)
        
        #get params
        mapx=headerdict['File Header']['Xres']  #map dimension x
        mapy=headerdict['File Header']['Yres']  #map dimension y

    #assign map size based on dimensions
    totalpx=mapx*mapy     

    #print run params
    print(f"header length: {headerlen} (bytes)")
    print(f"map dimensions: {mapx} x {mapy}")

    #   if we are skipping some of the file
    #       assign the ratio and adjust totalpx
    if config.SHORTRUN:
        skipratio=config.shortpct/100
        trunc_y=int(np.ceil(mapy*skipratio))
        totalpx=mapx*trunc_y
        print(f"SHORT RUN: ending at {skipratio*100} %")

    print(f"pixels expected: {totalpx}")
    print("---------------------------")

    if config.FORCEREAD:
        #assign starting pixel index 
        idx=headerlen+2 #legnth of header + 2 bytes

        #initialise pixel param arrays
        pxlen=np.zeros(totalpx,dtype=np.uint16)
        xidx=np.zeros(totalpx,dtype=np.uint16)
        yidx=np.zeros(totalpx,dtype=np.uint16)
        det=np.zeros(totalpx,dtype=np.uint16)
        dt=np.zeros(totalpx,dtype=np.uint16)
        
        if config.DOCOLOURS == True:
            #initalise pixel colour arrays
            rvals=np.zeros(totalpx)
            gvals=np.zeros(totalpx)
            bvals=np.zeros(totalpx)
            totalcounts=np.zeros(totalpx)

        #initialise data array
        data=np.zeros((totalpx,config.NCHAN),dtype=np.uint16)

        i=0 #pixel counter
        j=0 #row counter

        #loop through pixels
        while idx < streamlen:

            #print pixel index every row px
            if i % mapx == 0: 
                print(f"Row {j}/{mapy} at pixel {i}, byte {idx} ({100*idx/streamlen:.1f} %)", end='\r')
                j+=1

            #read pixel record into spectrum and header param arrays, 
            # + reassign index at end of read
            outchan, counts, pxlen[i], xidx[i], yidx[i], det[i], dt[i], idx = bitops.readpxrecord(idx, stream)

            #fill gaps in spectrum 
            #   (ie. assign all zero-count chans = 0)
            outchan, counts = utils.gapfill(outchan,counts, config.NCHAN)

            #warn if recieved channel list is different length to chan array
            if len(outchan) != len(chan):
                print("WARNING: unexpected length of channel list")
        
            #assign counts into data array - 
            data[i,:]=counts

            #build colours if required
            if config.DOCOLOURS == True: rvals[i], bvals[i], gvals[i], totalcounts[i] = colour.spectorgb(energy, counts)
            
            #if pixel index greater than expected no. pixels based on map dimensions
            #   end if we are doing a truncated run
            #   else throw a warning
            if i > (totalpx-1):
                if (config.SHORTRUN == True):   #i > totalpx is expected for short run
                    print("ending at:", idx)
                    idx=streamlen+1
                    break 
                else:
                    print(f"WARNING: pixel count {i} exceeds expected map size {totalpx}")
            i+=1

        nrows=j #store no. rows read successfully

        runtime = time.time() - starttime

        if config.SAVEPXSPEC:
            print(f"saving spectrum-by-pixel to file")
            np.savetxt(os.path.join(config.odir,  config.savename + ".dat"), data, fmt='%i')
        
        np.savetxt(os.path.join(config.odir, "pxlen.txt"), pxlen, fmt='%i')
        np.savetxt(os.path.join(config.odir, "xidx.txt"), xidx, fmt='%i')
        np.savetxt(os.path.join(config.odir, "yidx.txt"), yidx, fmt='%i')
        np.savetxt(os.path.join(config.odir, "detector.txt"), det, fmt='%i')
        np.savetxt(os.path.join(config.odir, "dt.txt"), dt, fmt='%i')


        print(
        "---------------------------\n"
        "MAP COMPLETE\n"
        "---------------------------\n"
        f"pixels expected (X*Y): {totalpx}\n"
        f"pixels found: {i}\n"
        f"total time: {round(runtime,2)} s\n"
        f"time per pixel: {round((runtime/i),6)} s\n"
        "---------------------------"
    )
    else:
        print("loading from file", config.savename)
        data = np.loadtxt(os.path.join(config.odir, config.savename), dtype=np.uint16)
        pxlen=np.loadtxt(os.path.join(config.odir, "pxlen.txt"), dtype=np.uint16)
        xidx=np.loadtxt(os.path.join(config.odir, "xidx.txt"), dtype=np.uint16)
        yidx=np.loadtxt(os.path.join(config.odir, "yidx.txt"), dtype=np.uint16)
        det=np.loadtxt(os.path.join(config.odir, "detector.txt"), dtype=np.uint16)
        dt=np.loadtxt(os.path.join(config.odir, "dt.txt"), dtype=np.uint16)
        print("loaded successfully", config.savename)
    "---------------------------\n"
    "Memory usage:\n"
    "---------------------------\n"
    utils.varsizes(locals().items())

    #clear the bytestream from memory
    del stream
    gc.collect()

    if config.DOCOLOURS == True:
        rgbarray=colour.clcomplete(rvals, gvals, bvals, totalcounts, mapx, nrows)
        colour.clshow(rgbarray)

    print("DOCLUST", config.DOCLUST)
    if config.DOCLUST:
        embedding, clusttimes = clustering.reduce(data)
        categories = clustering.dokmeans(embedding, totalpx)

        clustaverages=np.zeros([len(clustering.reducers),config.nclust,config.NCHAN])

        for i in range(len(clustering.reducers)):
            redname=clustering.getredname(i)
            clustaverages[i]=clustering.sumclusters(data, categories[i])
            
            for j in range(config.nclust):
                print(f'saving reducer {redname} cluster {j} with shape {clustaverages[i,j,:].shape}')
                np.savetxt(os.path.join(config.odir, "sum_" + redname + "_" + str(j) + ".txt"), np.c_[energy, clustaverages[i,j,:]], fmt=['%1.3e','%1.6e'])
            
            print(f'saving combined file for {redname}')
            np.savetxt(os.path.join(config.odir, "sum_" + redname + ".txt"), np.c_[energy, clustaverages[i,:,:].transpose(1,0)], fmt='%1.5e')             
            #plt.plot(energy, clustaverages[i,j,:])
        clustering.clustplt(embedding, categories, mapx, clusttimes)


print("CLEAN EXIT")
exit()


"""
snip background
https://stackoverflow.com/questions/57350711/baseline-correction-for-spectroscopic-data
"""