import struct 
import os
import numpy as np
import json
import copy

import src.utils as utils
import src.colour as colour
import src.fitting as fitting
import src.byteops as byteops

#-----------------------------------
#CLASSES
#-----------------------------------

#CLASSES
class Map:
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
        self.chunkidx = self.idx

        #read the JSON header and move pointer to start of first px record
        self.idx, self.headerdict = readgpxheader(config, self)
        
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

        if config['DOSUBMAP']:
            writegpxheader(config, self)

    def parse(self, config, pixelseries):
        """
        parse the pixel records from .GeoPIXE file
        takes stream of bytes, header length, chan/emap

        """
        print(f"pixels expected: {self.numpx}")
        print("---------------------------")

        #loop through pixels
        #while self.fullidx < self.fullsize:
        #while self.idx < self.streamlen:
        while True:
            #read pixel record into spectrum and header param arrays, 
            # + reassign index at end of read
            outchan, counts = readpxrecord(config, self, pixelseries)

            #fill gaps in spectrum 
            #   (ie. assign all zero-count chans = 0)
            outchan, counts = utils.gapfill(outchan,counts, config['NCHAN'])

            #warn if recieved channel list is different length to chan array
            if len(outchan) != len(self.chan):
                print("WARNING: unexpected length of channel list")

            #assign counts into data array
            if not config['SUBMAPONLY']:
                pixelseries.data[self.pxidx,:]=counts

            #if we are attempting to fit a background
            #   apply it, and save the corrected spectra
            if config['DOBG']: 
                counts, bg = fitting.fitbaseline(counts, config['LOWBGADJUST'])
                pixelseries.corrected[self.pxidx,:]=counts

            #build colours if required
            if config['DOCOLOURS'] == True: 
                pixelseries.rvals[self.pxidx], pixelseries.bvals[self.pxidx], pixelseries.gvals[self.pxidx], pixelseries.totalcounts[self.pxidx] = colour.spectorgb(config, self.energy, counts)
            

            #if pixel index greater than expected no. pixels based on map dimensions
            #   end if we are doing a truncated run
            #   else throw a warning
            if self.pxidx == (self.numpx-1):
                if (config['SHORTRUN'] == True):   #i > totalpx is expected for short run
                    print("short run ending at:", self.pxidx, self.idx)
                    self.idx=self.fullsize+1
                    break 
                else:
                    print(f"ENDING AT: Row {self.rowidx}/{self.yres} at pixel {self.pxidx}")
                    break
            elif self.pxidx > (self.numpx-1):
                print(f"WARNING: pixel count {self.pxidx} exceeds expected length: {self.numpx-1}")
                break

            #print pixel index every row
            #incrementing is skipped on final px
            if self.pxidx % self.xres == (self.xres-1): 
                self.fullidx=self.chunkidx+self.idx
                print(f"Row {self.rowidx}/{self.yres-1} at pixel {self.pxidx}, byte {self.fullidx} ({100*self.fullidx/self.fullsize:.1f} %)", end='\r')
                self.rowidx+=1
            self.pxidx+=1    #next pixel
        
        #store no. pixels and rows read successfully
        pixelseries.npx=self.pxidx+1
        pixelseries.nrows=self.rowidx+1 

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
    def __init__(self, config, map):
        #initialise pixel value arrays
        self.pxlen=np.zeros(map.numpx,dtype=np.uint16)
        self.xidx=np.zeros(map.numpx,dtype=np.uint16)
        self.yidx=np.zeros(map.numpx,dtype=np.uint16)
        self.det=np.zeros(map.numpx,dtype=np.uint16)
        self.dt=np.zeros(map.numpx,dtype=np.uint16)

        #create colour-associated attrs even if not doing colours
        self.rvals=np.zeros(map.numpx)
        self.gvals=np.zeros(map.numpx)
        self.bvals=np.zeros(map.numpx)
        self.totalcounts=np.zeros(map.numpx)

        #initialise whole data containers (WARNING: large)
        if not config['SUBMAPONLY']: 
            self.data=np.zeros((map.numpx,config['NCHAN']),dtype=np.uint16)
            if config['DOBG']: self.corrected=np.zeros((map.numpx,config['NCHAN']),dtype=np.uint16)
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


def readgpxheader(config, map):
    """
    read header 
        receives stream
        returns
            mapx
            mapy
            totalpx?
    """

    print(
        "---------------------------\n"
        f"PARSING HEADER\n"
        "---------------------------"
    )

    streamlen=len(map.stream)
    print(f"filesize: {streamlen} (bytes)")
    #if beginning of file
    #   read header length from first bytes as <uint16
    if map.idx == 0:
        headerlen=byteops.binunpack(map,"<H")
    else:
        raise ValueError("FATAL: attempting to read file header from nonzero index")

    #check for header
    #   pixels start with "DP" (=20550 as <uint16)
    #   if we find this immediately, header is zero length - cannot proceed
    #provided header is present
    #   read params from header
    if headerlen == 20550:  #(="DP" as <uint16)
        raise ValueError("FATAL: file header missing, cannot read map params")
    #also fail if headerlength is below arbitrary value
    elif headerlen <= 500:
        raise ValueError("FATAL: file header too small, check input")
    #else proceed
    else:
        """
        if header present, read as json
        https://stackoverflow.com/questions/40059654/python-convert-a-bytes-array-into-json-format
        """
        #pull slice of byte stream corresponding to header
        #   bytes[0-2]= headerlen
        #   doesn't include trailing '\n' '}', so +2
        headerraw=map.stream[2:headerlen+2]

        #read it as utf8
        headerstream = headerraw.decode('utf-8')
        
        #load into dictionary via json builtin
        headerdict = json.loads(headerstream)

    #print map params
    print(f"header length: {headerlen} (bytes)")

    #set pointer index to length of header + 2 bytes
    idx = headerlen+2

    return idx, headerdict

def writegpxheader(config, map):
    #modify width and height in header and re-print

    newxres=config['submap_x2']-config['submap_x1']
    #if new res larger than original, set to original
    if newxres > map.xres:
        newxres = map.xres
    newxdim=newxres*(map.headerdict["File Header"]["Width (mm)"]/map.headerdict["File Header"]["Xres"])

    newyres=config['submap_y2']-config['submap_y1']
    #if new res larger than original, set to original
    if newyres > map.yres:
        newyres = map.yres
    newydim=newyres*(map.headerdict["File Header"]["Height (mm)"]/map.headerdict["File Header"]["Yres"])

    #create a duplicate via deepcopy
    #   need deepcopy because nested lists - normal copy would point to original data still
    newheaderdict = copy.deepcopy(map.headerdict)
    newheaderdict["File Header"]["Xres"]=newxres
    newheaderdict["File Header"]["Width (mm)"]=newxdim
    newheaderdict["File Header"]["Yres"]=newyres
    newheaderdict["File Header"]["Height (mm)"]=newydim

    #create a printable version  
    headerdump = json.dumps(newheaderdict, indent='\t', sort_keys=False)
    #create a byte-encoded version 
    headerencode = headerdump.encode('utf-8')

    #write the new header length
    map.outfile.write(struct.pack("<H",len(headerencode)))
    #write the new header
    map.outfile.write(headerencode)

    #NB: PROBLEM HERE ----------------
    # The default JSON has a duplicate entry.
    # "Detector" appears twice beacuse there are two dets
    # first is overwritten during json.loads
    #   .: only one in dict to write to second file
    #   think we can ignore this, the info is not used, but header is different when rewritten



def readpxrecord(config, map, pixelseries):
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
    pxheaderlen=config['PXHEADERLEN']   #create a local var for readability
    pxstart=map.idx
#   check for pixel start flag "DP" at first position after header:
    #   unpack first two bytes after header as char
    pxflag=struct.unpack("cc", map.stream[map.idx:map.idx+2])[:]

    #   use join to merge into string
    pxflag="".join([pxflag[0].decode(config['CHARENCODE']),pxflag[1].decode(config['CHARENCODE'])])

    #   check if string is "DP" - if not, fail
    if pxflag != config['PXFLAG']:
       raise ValueError(f"ERROR: pixel flag 'DP' expected but not found at byte {map.idx}")

    map.idx=map.idx+2   #step over "DP"

    #read each header field and step idx to end of field
    pxlen=byteops.binunpack(map,"<I")
    xcoord=byteops.binunpack(map,"<H")
    ycoord=byteops.binunpack(map,"<H")
    det=byteops.binunpack(map,"<H")
    dt=byteops.binunpack(map,"<f")
    #   faster to unpack into temp variables vs directly into pbject attrs. not sure why atm

    #initialise channel index and result arrays
    j=0 #channel index
    chan=np.zeros(int((pxlen-pxheaderlen)/config['BYTESPERCHAN']), dtype=int)
    counts=np.zeros(int((pxlen-pxheaderlen)/config['BYTESPERCHAN']), dtype=int)
    #       4 = no. bytes in each x,y pair
    #         = 2x2 bytes each 

    #if writing and coordinates within subregion specified in config
    if (config['DOSUBMAP'] and
            xcoord >= config['submap_x1'] and xcoord < config['submap_x2'] and
            ycoord >= config['submap_y1'] and ycoord < config['submap_y2']
    ):
            #raise an error if chunk breaks pixel header - should be rare
        if pxstart+pxheaderlen > map.streamlen:
            raise ValueError("ERROR: pixel header crosses chunk - increase chunk size slightly and repeat")

        #   export the pixel header
        #save the header as series of bytes
        #   modifying start coords to new structure (ie. px1 now 0,0)

        outstream=struct.pack("!c",pxflag[0].encode('ascii'))
        outstream+=struct.pack("!c",pxflag[1].encode('ascii'))
        #   https://stackoverflow.com/questions/33521838/python-struct-sending-chars
        outstream+=struct.pack("<I",pxlen)
        outstream+=struct.pack("<H",xcoord-config['submap_x1'])
        outstream+=struct.pack("<H",ycoord-config['submap_y1'])
        outstream+=struct.pack("<H",det)
        outstream+=struct.pack("<f",dt)

        map.outfile.write(outstream)
        map.outfile.write(map.stream[pxstart+pxheaderlen:pxstart+pxlen])

    if config['SUBMAPONLY']:
        #if writing only, push pointer forward to next pixel record
        #   (ie. increase by pxlen, backtrack by fixed px header length)
        if not pxstart+pxlen >= map.streamlen:    #provided we are not near the end of a chunk
            map.idx=pxstart+pxlen
        else:   #if step would exceed chunk
            part=map.streamlen-pxstart  #store the length remaining
            map.nextchunk()                    #load next
            map.idx=(pxlen-part)        #push pointer by remainder
    else:
        #iterate through channel/count pairs 
        #   until byte index passes pxlen
        while j*config['BYTESPERCHAN'] < pxlen-pxheaderlen:
            chan[j]=byteops.binunpack(map,"<H")
            counts[j]=byteops.binunpack(map,"<H")
            j+=1    #next channel

    #assign object attrs from temp vars
    pixelseries.pxlen[map.pxidx]=pxlen
    pixelseries.xidx[map.pxidx]=xcoord
    pixelseries.yidx[map.pxidx]=ycoord
    pixelseries.det[map.pxidx]=det
    pixelseries.dt[map.pxidx]=dt

    return(chan, counts)

def readspec(config, odir):
    """
    read data from a pre-saved datfile
        does not currently return as much information as the full parse
    """
    print("loading from file", config['outfile'])
    data = np.loadtxt(os.path.join(odir, config['outfile']), dtype=np.uint16)
    pxlen=np.loadtxt(os.path.join(odir, "pxlen.txt"), dtype=np.uint16)
    xidx=np.loadtxt(os.path.join(odir, "xidx.txt"), dtype=np.uint16)
    yidx=np.loadtxt(os.path.join(odir, "yidx.txt"), dtype=np.uint16)
    det=np.loadtxt(os.path.join(odir, "detector.txt"), dtype=np.uint16)
    dt=np.loadtxt(os.path.join(odir, "dt.txt"), dtype=np.uint16)
    print("loaded successfully", config['outfile']) 

    corrected=None
    rvals=None
    bvals=None
    gvals=None
    totalcounts=None
    nrows=None

    return(data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows) 