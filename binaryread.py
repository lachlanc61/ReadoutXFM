import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import cv2
import os
import glob
from decimal import *
from scipy.optimize import curve_fit


#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
#global variables
FTYPE=".GeoPIXE"    #valid: ".GeoPIXE"
DEBUG=False     #debug flag 

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
infile = "leaf2_overview.GeoPIXE"    #assign input file
                                #   only used if reading .geo

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

#initialise plot defaults
plt.rc('font', size=smallfont)          # controls default text sizes
plt.rc('axes', titlesize=smallfont)     # fontsize of the axes title
plt.rc('axes', labelsize=medfont)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
plt.rc('legend', fontsize=smallfont)    # legend fontsize
plt.rc('figure', titlesize=lgfont)      # fontsize of the figure title
plt.rc('lines', linewidth=lwidth)
plt.rcParams['axes.linewidth'] = bwidth

#-----------------------------------
#MAIN START
#-----------------------------------

#read in either .avi or .tif files
#   paired with if/else at beginning of frame-by-frame read
#   clunky but best I can think of so far
#       some possibility to have orphaned variables - eg. vidcap doesn't exist if filetype is tif

#if filetype is avi, read frame-by-frame from avi
if FTYPE == ".GeoPIXE":
    f = os.path.join(wdir,infile)
    fname = os.path.splitext(os.path.basename(f))[0]

    print("opening .geo:",fname)
else: 
    print(f'FATAL: filetype {FTYPE} not recognised')
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