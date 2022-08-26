import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import cv2
import os
import glob
import struct 
from decimal import *
from scipy.optimize import curve_fit


#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
#global variables
FTYPE=".GeoPIXE"    #valid: ".GeoPIXE"
DEBUG=False     #debug flag 
PXHEADERLEN=16  #pixel header size
PXFLAG="DP"
NCHAN=4096
CHARENCODE = 'utf-8'

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
infile = "leaf2_overview.GeoPIXE"    #assign input file
                                #   only used if reading .geo

detid="A"

#figure params
colourmap='Set1'    #colourmap for figure
#colourmap='copper'    #colourmap for figure
figx=20         #cm width of figure
figy=10         #cm height of figure
smallfont = 8   #default small font
medfont = 10    #default medium font
lgfont = 12     #default large font
lwidth = 1  #default linewidth
bwidth = 1  #default border width


#-------------------------------------
#FUNCTIONS
#-----------------------------------

#-----------------------------------
#INITIALISE
#-----------------------------------

#initialise directories relative to script
script = os.path.realpath(__file__) #_file = current script
spath=os.path.dirname(script) 
wdir=os.path.join(spath,wdirname)
odir=os.path.join(spath,odirname)
print("script:", script)
print("script path:", spath)
print("data path:", wdir)
print("---------------")
#-----------------------------------
#MAIN START
#-----------------------------------
"""
Read binary with struct
https://stackoverflow.com/questions/8710456/reading-a-binary-file-with-python
Read binary as chunks
https://stackoverflow.com/questions/71978290/python-how-to-read-binary-file-by-chunks-and-specify-the-beginning-offset
"""

if FTYPE == ".GeoPIXE":
    f = os.path.join(wdir,infile)
    fname = os.path.splitext(os.path.basename(f))[0]

    print("opening .geo:",fname)
else: 
    print(f'FATAL: filetype {FTYPE} not recognised')
    exit()

print("---------------")

with open(f, mode='rb') as file: # rb = read binary
    
    stream = file.read()    #NB. to read in chunks, add chunk size as read(SIZE)
    print("stream length in bytes ",len(stream))

    print("first two bytes: ",stream[:2])

    #struct unpack outputs tuple, want int
    #get header length from first two bytes
    #   read as little-endian uint16 "<H"
    headerlen = int(struct.unpack("<H", stream[:2])[0])
    print(f"header length: {headerlen}")

    #occasionally, files might be missing the header
    #   when this happens, the first bytes are "DP" - denoting the start of a pixel record
    #   therefore if we get 20550 (="DP" as <uint16), header is missing
    if headerlen == 20550:
        print("WARNING: no header found")
        headerlen=0

#   look for pixel start flag "DP" at first position after header:
    #   unpack first two bytes after header as char
    pxflag=struct.unpack("cc", stream[headerlen+2:headerlen+4])[:]
    #   use join to merge into string
    pxflag="".join([pxflag[0].decode(CHARENCODE),pxflag[1].decode(CHARENCODE)])
    #   check if string is "DP" - if not, fail
    if pxflag != PXFLAG:
        print(f"ERROR: pixel flag 'DP' not found at byte {headerlen+2}")
        exit()
    else:
        print(f"pixel at byte {headerlen+2}")

    print("ints: ",stream[headerlen+14:headerlen+20])
    print(stream[headerlen+14:headerlen+15])

    #initialise spectrum arrays
    j=0
    kv=np.zeros((NCHAN), dtype=float)
    counts=np.zeros((NCHAN), dtype=int)
    for i in np.arange(headerlen+2+PXHEADERLEN, headerlen+2+PXHEADERLEN+NCHAN, 4):
        kv[j]=0.01*(struct.unpack("<H", stream[i:i+2])[0])
        counts[j]=int(struct.unpack("<H", stream[i+2:i+4])[0])
        print(round(kv[j],2),counts[j])
    
    """"
    := c + pixHeaderLength; i+4 < pixelDataEnd; {
                        channel := binary.LittleEndian.Uint16(bytes[i : i+2])
                        count := binary.LittleEndian.Uint16(bytes[i+2 : i+4])
    """
exit()

































success=True
steps=np.arange(10)
while success:
    #initialise plot and colourmaps per frame
    plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
    plt.rcParams["figure.figsize"] = [figx/2.54, figy/2.54]
    fig=plt.figure()
    lut = cm = plt.get_cmap(colourmap) 
    cNorm  = colors.Normalize(vmin=0, vmax=len(steps)+2)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=lut)

    #READ NEXT FRAME
    #filetype switcher again - read .avi and .tif differently
    #   paired with switcher at top of main 
    #   - be very careful that all branches send the same results downstream
    if FTYPE == ".GeoPIXE":
        #read a frame from the avi, returning success
        readimage = f

        if readimage is not None:
                success=True
        else:
            print("failed for",f)
            success=False
    else:
        print("FATAL: filetype {%} not recognised",FTYPE)
        exit()    

    print('Read frame : ', readimage)

    #leave loop if import unsuccessful (ie. no more frames)
    if not success:
        break
    
    exit()
print("CLEAN END")