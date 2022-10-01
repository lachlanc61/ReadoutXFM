import config
import struct 
import numpy as np

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

    #struct unpack outputs tuple
    #want int so take first value
    retval = struct.unpack(sformat, stream[idx:idx+nbytes])[0]
    idx=idx+nbytes
    return(retval, idx)    

def readpxrecord(idx, stream):
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
    pxstart=idx
#   check for pixel start flag "DP" at first position after header:
    #   unpack first two bytes after header as char
    pxflag=struct.unpack("cc", stream[idx:idx+2])[:]
    #   use join to merge into string
    pxflag="".join([pxflag[0].decode(config.CHARENCODE),pxflag[1].decode(config.CHARENCODE)])

    #   check if string is "DP" - if not, fail
    if pxflag != config.PXFLAG:
        print(f"ERROR: pixel flag 'DP' expected but not found at byte {idx}")
        exit()
    else:
        if (config.DEBUG): print(f"pixel at: {idx} bytes")

    idx=idx+2   #step over "DP"
    if (config.DEBUG): print(f"next bytes at {idx}: {stream[idx:idx+config.PXHEADERLEN]}")
    #read each header field and step idx to end of field
    pxlen, idx=binunpack(stream,idx,"<I")
    xcoord, idx=binunpack(stream,idx,"<H")
    ycoord, idx=binunpack(stream,idx,"<H")
    det, idx=binunpack(stream,idx,"<H")
    dt, idx=binunpack(stream,idx,"<f")

    #print header fields
    if (config.DEBUG): 
        print("PXLEN: ",pxlen)
        print("XCOORD: ",xcoord)
        print("YCOORD: ",ycoord)
        print("DET: ",det)
        print("DT: ",dt)

    #initialise channel index and result arrays
    j=0
    chan=np.zeros(int((pxlen-config.PXHEADERLEN)/4), dtype=int)
    counts=np.zeros(int((pxlen-config.PXHEADERLEN)/4), dtype=int)
    #       4 = no. bytes in each x,y pair
    #         = 2x2 bytes each 

    #iterate until byte index passes pxlen
    #pull channel, count pairs
    while idx < (pxstart+pxlen):
    #while idx < 2000:
        if (config.DEBUG2): print(f"next bytes at {idx}: {stream[idx:idx+8]}")
        chan[j], idx=binunpack(stream,idx,"<H")
        counts[j], idx=binunpack(stream,idx,"<H")
        if (config.DEBUG2): print(f"idx {idx} x {chan[j]} y {counts[j]}")
        
    #    if (DEBUG): print(f"idx {idx} / {pxstart+pxlen}")
        j=j+1
    if (config.DEBUG): print(f"following bytes at {idx}: {stream[idx:idx+10]}")
    return(chan, counts, pxlen, xcoord, ycoord, det, dt, idx)