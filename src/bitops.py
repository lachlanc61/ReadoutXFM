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
        try:
            self.infile = open(fi, mode='rb') # rb = read binary
            self.outfile = open(fo, mode='wb')   #wb = write binary
        except FileNotFoundError:
            print("FATAL: incorrect filepath/files not found")

        #get total size of file to parse
        self.fullsize = os.path.getsize(fi)
        self.chunksize = config['chunksize']

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
        self.idx, self.headerdict = readgpxheader(self)
        
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

        if config['DOWRITE']:
            self.outfile.write(self.stream[0:self.idx])

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

            #print pixel index every row px
            #incrementing is skipped on final px
            if self.pxidx % self.xres == (self.xres-1): 
                self.fullidx=self.chunkidx+self.idx
                print(f"Row {self.rowidx}/{self.yres} at pixel {self.pxidx}, byte {self.fullidx} ({100*self.fullidx/self.fullsize:.1f} %)", end='\r')
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

    def next(self):
        self.chunkidx = self.chunkidx + self.idx

        self.stream = self.infile.read(self.chunksize)

        if len(self.stream) != self.streamlen:
            print("WARNING: final stream")

        self.streamlen=len(self.stream)
        self.idx=0

        if not self.stream:
            print("no stream found")
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

def binunpack(map, sformat):
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

    #if perfect end
    #   unpack then flag next chunk
    if map.idx == (len(map.stream)-nbytes):
        retval = struct.unpack(sformat, map.stream[map.idx:map.idx+nbytes])[0]
        map.idx=map.idx+nbytes
        map.next()
    #if end mid unpack
    #   unpack partial, get next chunk, unpack next partial and concat
    elif map.idx > (len(map.stream)-nbytes):
        remaining=(len(map.stream)-map.idx)
        partial1 = map.stream[map.idx:map.idx+remaining]
        map.idx=map.idx+remaining
        map.next()
        partial2 = map.stream[map.idx:map.idx+(nbytes-remaining)]
        map.idx=map.idx+(nbytes-remaining)
        #concat partials
        partial=partial1+partial2
        #WARNING: DOES NOT WORK YET
        # aim is to concat bytes, apparently don't behave like strings
        # need to convert to bytearray or similar
        # https://stackoverflow.com/questions/28130722/python-bytes-concatenation
        retval=struct.unpack(sformat, partial)[0]

    #if not at end
    #   unpack and increment index
    else:
        #struct unpack outputs tuple
        #want int so take first value
        retval = struct.unpack(sformat, map.stream[map.idx:map.idx+nbytes])[0]
        map.idx=map.idx+nbytes

    return(retval)



def readgpxheader(map):
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
        headerlen=binunpack(map,"<H")
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
        #   headerlen doesn't include trailing '\n' '}', so +2
        headerstream=map.stream[2:headerlen+2]
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

    pxstart=map.idx
#   check for pixel start flag "DP" at first position after header:
    #   unpack first two bytes after header as char
    pxflag=struct.unpack("cc", map.stream[map.idx:map.idx+2])[:]

    #   use join to merge into string
    pxflag="".join([pxflag[0].decode(config['CHARENCODE']),pxflag[1].decode(config['CHARENCODE'])])

    #   check if string is "DP" - if not, fail
    if pxflag != config['PXFLAG']:
        print(f"ERROR: pixel flag 'DP' expected but not found at byte {map.idx}")
        exit()

    map.idx=map.idx+2   #step over "DP"

    #read each header field and step idx to end of field
    pxlen=binunpack(map,"<I")
    xcoord=binunpack(map,"<H")
    ycoord=binunpack(map,"<H")
    det=binunpack(map,"<H")
    dt=binunpack(map,"<f")
    #   faster to unpack into temp variables vs directly into pbject attrs. not sure why atm

    #initialise channel index and result arrays
    j=0 #channel index
    chan=np.zeros(int((pxlen-config['PXHEADERLEN'])/config['BYTESPERCHAN']), dtype=int)
    counts=np.zeros(int((pxlen-config['PXHEADERLEN'])/config['BYTESPERCHAN']), dtype=int)
    #       4 = no. bytes in each x,y pair
    #         = 2x2 bytes each 

    #if writing and coordinates within subregion specified in config
    if (config['DOWRITE'] and
            xcoord >= config['writestartx'] and xcoord < config['writeendx'] and
            ycoord >= config['writestarty'] and ycoord < config['writeendy']
        ):
            #export this pixel
            map.outfile.write(map.stream[pxstart:pxstart+pxlen])

    if config['WRITEONLY']:
        #if writing only, push pointer forward to next pixel record
        #   (ie. increase by pxlen, backtrack by fixed px header length)
        map.idx=pxstart+pxlen
    else:
        #iterate through channel/count pairs 
        #   until byte index passes pxlen
        while j*config['BYTESPERCHAN'] < pxlen-config['PXHEADERLEN']:
            chan[j]=binunpack(map,"<H")
            counts[j]=binunpack(map,"<H")
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