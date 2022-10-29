import struct 
import os
import numpy as np
import json
import copy

import src.utils as utils
import src.colour as colour
import src.fitting as fitting
import src.byteops as byteops
import src.parser as parser


#CLASSES
class Xfmap:
    def __init__(self, config, fi, fo):

        #assign input file object for reading
        try:
            self.infile = open(fi, mode='rb') # rb = read binary
            self.outfile = open(fo, mode='wb')   #wb = write binary
        except FileNotFoundError:
            print("FATAL: incorrect filepath/files not found")

        #get total size of file to parse
        self.fullsize = os.path.getsize(fi)
        self.chunksize = int(config['chunksize'])

        #generate initial bytestream
        #self.stream = self.infile.read()         
        self.stream = self.infile.read(self.chunksize)   
        self.streamlen=len(self.stream)

        #pointers
        self.idx=0      #byte pointer
        self.pxidx=0    #pixel pointer
        self.rowidx=0   #row pointer
        self.pxstart=0  #pointer for start of pixel

        self.fullidx = self.idx
        self.chunkidx = self.idx

        #read the JSON header and move pointer to start of first px record
        self.idx, self.headerdict = parser.readfileheader(config, self)
        
        #try to assign values from header
        try:
            self.xres = self.headerdict['File Header']['Xres']           #map size x
            self.yres = self.headerdict['File Header']['Yres']           #map size y
            self.xdim = self.headerdict['File Header']['Width (mm)']     #map dimension x
            self.ydim = self.headerdict['File Header']['Height (mm)']    #map dimension y
            self.nchannels = int(self.headerdict['File Header']['Chan']) #no. channels
            self.gain = float(self.headerdict['File Header']['Gain (eV)']/1000) #gain in kV
        except:
            raise ValueError("FATAL: failure reading values from header")
        
        #initialise arrays
        self.chan = np.arange(0,self.nchannels)     #channel series
        self.energy = self.chan*self.gain           #energy series
        self.xarray = np.arange(0, self.xdim, self.xdim/self.xres )   #position series x  
        self.yarray = np.arange(0, self.ydim, self.ydim/self.yres )   #position series y
            #NB: real positions likely better represented by centres of pixels eg. 0+(xdim/xres), xdim-(xdim/xres) 
            #       need to ask IXRF how this is handled by Iridium

        #derived vars
        self.numpx = self.xres*self.yres        #expected number of pixels

        #init struct for reading pixel headers
        self.headstruct=struct.Struct("<3Hf")
        self.minstruct=struct.Struct("<ccI")
        self.PXHEADERLEN=config['PXHEADERLEN'] 
        self.pxlen=self.PXHEADERLEN+config['NCHAN']*4   #dummy value for pxlen

    def parse(self, config, pxseries):

        print(f"pixels expected: {self.numpx}")
        print("---------------------------")

        if config['WRITESUBMAP']:
            parser.writefileheader(config, self)

        while True:
            
            pxlen = parser.initpx(config, self)
            
            pxseries.pxlen[self.pxidx]=pxlen

            locstream, self.idx = parser.getstream(self,self.idx,pxlen)

            pxseries = parser.readpxheader(locstream, config, pxseries)

            if config['WRITESUBMAP']:
                parser.writepxheader(config, self, pxseries)
                parser.writepxrecord(locstream, self.pxlen, self)

            if config['PARSEMAP']:
                chan, counts = parser.readpxdata(config, locstream, self.pxlen, self, pxseries)

                #fill gaps in spectrum 
                #   (ie. assign all zero-count chans = 0)
                chan, counts = utils.gapfill(chan,counts, config['NCHAN'])

                #warn if recieved channel list is different length to chan array
                if len(chan) != len(self.chan):
                    print("WARNING: unexpected length of channel list")

                #assign counts into data array
                pxseries.data[self.pxidx,:]=counts

            self.idx+=pxlen
            self.fullidx=self.chunkidx+self.idx

            #stop when pixel index greater than expected no. pixels
            if self.pxidx >= (self.numpx-1):
                print(f"ENDING AT: Row {self.rowidx}/{self.yres} at pixel {self.pxidx}")
                break

            #print pixel index at end of every row
            if self.pxidx % self.xres == (self.xres-1): 
                print(f"Row {self.rowidx}/{self.yres-1} at pixel {self.pxidx}, byte {self.fullidx} ({100*self.fullidx/self.fullsize:.1f} %)", end='\r')
                self.rowidx+=1

            self.pxidx+=1    #next pixel

        #store no. pixels and rows read successfully
        pxseries.npx=self.pxidx+1
        pxseries.nrows=self.rowidx+1 

    def read(self, config, odir):
        pass
        """
            data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows \
            = readspec(config, odir)
        """

    def nextchunk(self):
        self.chunkidx = self.chunkidx + self.idx

        self.stream = self.infile.read(self.chunksize)

        if len(self.stream) != self.streamlen:
            print("NOTE: final chunk")

        self.streamlen=len(self.stream)
        self.idx=0

        if not self.stream:
            print("NOTE: no chunks remaining")
            #exit()

    def closefiles(self):
        self.infile.close()
        self.outfile.close()


class PixelSeries:
    def __init__(self, config, xfmap):
        #initialise pixel value arrays
        self.pxlen=np.zeros(xfmap.numpx,dtype=np.uint16)
        self.xidx=np.zeros(xfmap.numpx,dtype=np.uint16)
        self.yidx=np.zeros(xfmap.numpx,dtype=np.uint16)
        self.det=np.zeros(xfmap.numpx,dtype=np.uint16)
        self.dt=np.zeros(xfmap.numpx,dtype=np.uint16)

        #create colour-associated attrs even if not doing colours
        self.rvals=np.zeros(xfmap.numpx)
        self.gvals=np.zeros(xfmap.numpx)
        self.bvals=np.zeros(xfmap.numpx)
        self.totalcounts=np.zeros(xfmap.numpx)

        #initialise whole data containers (WARNING: large)
        if not config['SUBMAPONLY']: 
            self.data=np.zeros((xfmap.numpx,config['NCHAN']),dtype=np.uint16)
            if config['DOBG']: self.corrected=np.zeros((xfmap.numpx,config['NCHAN']),dtype=np.uint16)
        else:
            self.data=np.zeros((1024,config['NCHAN']),dtype=np.uint16)

        self.npx=0
        self.nrows=0


    def exportheader(self, config, odir):
        np.savetxt(os.path.join(odir, "pxlen.txt"), self.pxlen, fmt='%i')
        np.savetxt(os.path.join(odir, "xidx.txt"), self.xidx, fmt='%i')
        np.savetxt(os.path.join(odir, "yidx.txt"), self.yidx, fmt='%i')
        np.savetxt(os.path.join(odir, "detector.txt"), self.det, fmt='%i')
        np.savetxt(os.path.join(odir, "dt.txt"), self.dt, fmt='%i')

    def exportseries(self, config, odir):
        print(f"saving spectrum-by-pixel to file")
        np.savetxt(os.path.join(odir,  config['outfile'] + ".dat"), self.data, fmt='%i')
