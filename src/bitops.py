import struct 
import os
import numpy as np
import json

import src.utils as utils
import src.colour as colour
import src.fitting as fitting

#-----------------------------------
#CLASSES
#-----------------------------------

#CLASSES
class Map:
    def __init__(self, config, fi, fo):

        #assign input file object for reading
        self.infile = open(fi, mode='rb') # rb = read binary
        self.outfile = open(fo, mode='wb')   #wb = write binary

        #get total size of file to parse
        self.fullsize = os.path.getsize(fi)

        #generate initial bytestream
        self.stream = self.infile.read()         
        #stream = infile.read(config['chunksize'])   
        self.streamlen=len(self.stream)

        self.idx, self.headerdict = readgpxheader(self.stream)

        self.fullidx = self.idx

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

    def parse(self, config, pixelseries):
        """
        parse the pixel records from .GeoPIXE file
        takes stream of bytes, header length, chan/emap

        """
        print(f"pixels expected: {self.numpx}")
        print("---------------------------")

        idx=self.idx
        i=0    #pixel counter
        j=0 #row counter

        #loop through pixels
        #while self.fullidx < self.fullsize:
        while idx < self.streamlen:

            #print pixel index every row px
            if i % self.xdim == 0: 
                print(f"Row {j}/{self.ydim} at pixel {i}, byte {idx} ({100*self.fullidx/self.fullsize:.1f} %)", end='\r')
                j+=1

            #read pixel record into spectrum and header param arrays, 
            # + reassign index at end of read
            outchan, counts, pixelseries.pxlen[i], pixelseries.xidx[i], pixelseries.yidx[i], pixelseries.det[i], pixelseries.dt[i] = readpxrecord(config, self)

            #fill gaps in spectrum 
            #   (ie. assign all zero-count chans = 0)
            outchan, counts = utils.gapfill(outchan,counts, config['NCHAN'])

            #warn if recieved channel list is different length to chan array
            if len(outchan) != len(self.chan):
                print("WARNING: unexpected length of channel list")
        
            #assign counts into data array
            pixelseries.data[i,:]=counts

            #if we are attempting to fit a background
            #   apply it, and save the corrected spectra
            if config['DOBG']: 
                counts, bg = fitting.fitbaseline(counts, config['LOWBGADJUST'])
                pixelseries.corrected[i,:]=counts

            #build colours if required
            if config['DOCOLOURS'] == True: 
                pixelseries.rvals[i], pixelseries.bvals[i], pixelseries.gvals[i], pixelseries.totalcounts[i] = colour.spectorgb(config, self.energy, counts)
            
            #if pixel index greater than expected no. pixels based on map dimensions
            #   end if we are doing a truncated run
            #   else throw a warning
            if i > (self.numpx-2):
                if (config['SHORTRUN'] == True):   #i > totalpx is expected for short run
                    print("ending at:", i, idx)
                    idx=self.fullsize+1
                    break 
                else:
                    print(f"WARNING: pixel count {i} exceeds expected map size {self.numpx}")
            self.idx=idx
            i+=1    #next pixel
        pixelseries.npx=i
        pixelseries.nrows=j #store no. rows read successfully

    def read(self, config, odir):
        pass
        """
            data, corrected, pxlen, xidx, yidx, det, dt, rvals, bvals, gvals, totalcounts, nrows \
            = readspec(config, odir)
        """

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
        self.data=np.zeros((map.numpx,config['NCHAN']),dtype=np.uint16)
        if config['DOBG']: self.corrected=np.zeros((map.numpx,config['NCHAN']),dtype=np.uint16)

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

#-------------------------------------
#FUNCTIONS
#-----------------------------------

def binunpack(stream, idx, sformat):
    """
    parse binary data via struct.unpack
    takes:
        stream of bytes
        byte index
        format flag for unpack (currently accepts: <H <f <I )
    returns:
        value in desired format (eg. int, float)
        next byte index
    """

    if sformat == "<H":
        nbytes=2
    elif sformat == "<f":
        nbytes=4
    elif sformat == "<I":
        nbytes=4
    else:
        print(f"ERROR: {sformat} not recognised by local function binunpack")
        exit(0)

    """
    CHECK AND UPDATE STREAM and INDEX HERE
    """

    #struct unpack outputs tuple
    #want int so take first value
    retval = struct.unpack(sformat, stream[idx:idx+nbytes])[0]
    idx=idx+nbytes
    return(retval, idx)    



def readgpxheader(stream):
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

    streamlen=len(stream)
    print(f"filesize: {streamlen} (bytes)")

    headerlen=binunpack(stream,0,"<H")[0]

    #check for header
    #   pixels start with "DP" (=20550 as <uint16)
    #   if we find this immediately, header is zero length - cannot proceed
    #provided header is present
    #   read params from header

    if headerlen == 20550:  #(="DP" as <uint16)
        print("WARNING: no header found")
        headerlen=0
        print("FATAL: map dimensions unknown, cannot build map")
        exit()
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

    #print map params
    print(f"header length: {headerlen} (bytes)")

    #set pointer index to length of header + 2 bytes
    idx = headerlen+2

    return idx, headerdict


def readpxrecord(config, map):
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
    idx=map.idx
    stream=map.stream

    pxstart=idx

#   check for pixel start flag "DP" at first position after header:
    #   unpack first two bytes after header as char
    pxflag=struct.unpack("cc", stream[idx:idx+2])[:]

    #   use join to merge into string
    pxflag="".join([pxflag[0].decode(config['CHARENCODE']),pxflag[1].decode(config['CHARENCODE'])])

    #   check if string is "DP" - if not, fail
    if pxflag != config['PXFLAG']:
        print(f"ERROR: pixel flag 'DP' expected but not found at byte {idx}")
        exit()
    else:
        if (config['DEBUG']): print(f"pixel at: {idx} bytes")

    idx=idx+2   #step over "DP"

    if (config['DEBUG']): print(f"next bytes at {idx}: {stream[idx:idx+config['PXHEADERLEN']]}")

    #read each header field and step idx to end of field
    pxlen, idx=binunpack(stream,idx,"<I")
    xcoord, idx=binunpack(stream,idx,"<H")
    ycoord, idx=binunpack(stream,idx,"<H")
    det, idx=binunpack(stream,idx,"<H")
    dt, idx=binunpack(stream,idx,"<f")

    #initialise channel index and result arrays
    j=0 #channel index
    chan=np.zeros(int((pxlen-config['PXHEADERLEN'])/4), dtype=int)
    counts=np.zeros(int((pxlen-config['PXHEADERLEN'])/4), dtype=int)
    #       4 = no. bytes in each x,y pair
    #         = 2x2 bytes each 

    #iterate through channel/count pairs 
    #   until byte index passes pxlen
    while idx < (pxstart+pxlen):
        if (config['DEBUG2']): print(f"next bytes at {idx}: {stream[idx:idx+8]}")
        chan[j], idx=binunpack(stream,idx,"<H")
        counts[j], idx=binunpack(stream,idx,"<H")
        if (config['DEBUG2']): print(f"idx {idx} x {chan[j]} y {counts[j]}")
        j=j+1   #next channel
    if (config['DEBUG']): print(f"following bytes at {idx}: {stream[idx:idx+10]}")

    #update map positions
    map.idx = idx
    map.stream = stream

    return(chan, counts, pxlen, xcoord, ycoord, det, dt)

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