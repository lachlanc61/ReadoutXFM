import matplotlib.pyplot as plt
import os

#-----------------------------------
#USER MODIFIABLE CONSTANTS
#-----------------------------------
#global variables
FTYPE=".GeoPIXE"    #valid: ".GeoPIXE"

PXHEADERLEN=16  #pixel header size
PXFLAG="DP"
NCHAN=4096
ESTEP=0.01
CHARENCODE = 'utf-8'
DOCOLOURS=False
DOCLUST=True
CMAP='Set1'
#MAPX=128   #for leaf
#MAPY=68
MAPX=256    #for geo2
MAPY=126

#debug flags
DEBUG=False     #debug flag (pixels)
DEBUG2=False    #second-level debug flag (channels)
#shortcut flags
SHORTRUN=False  #stop after first X% of pixels
skipratio=5    #% of px

#recalc flags
FORCERED=True        #always recalc dimensionality reduction
FORCEKMEANS=True     #always recalc kmeans 

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
infile = "geo2.GeoPIXE"    #assign input file

#instrument config
detid="A"   #detector ID - not needed for single detector maps

#figure params
figx=20         #cm width of figure
figy=10         #cm height of figure
smallfont = 8   #default small font
medfont = 10    #default medium font
lgfont = 12     #default large font
lwidth = 1  #default linewidth
bwidth = 1  #default border width

#debug params
skipratio=20    #pct of file to finish at, if skipping

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


#plot defaults

if True:
    plt.rc('font', size=smallfont)          # controls default text sizes
    plt.rc('axes', titlesize=smallfont)     # fontsize of the axes title
    plt.rc('axes', labelsize=medfont)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=smallfont)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=smallfont)    # fontsize of the tick labels
    plt.rc('legend', fontsize=smallfont)    # legend fontsize
    plt.rc('figure', titlesize=lgfont)  # fontsize of the figure title
    plt.rc('lines', linewidth=lwidth)
    plt.rcParams['axes.linewidth'] = bwidth

"""
plot options for spectra - not currently used
fig=plt.figure()
ax=fig.add_subplot(111)
"""
"""
ax.set_yscale('log')

ax.set_ylabel('intensity (counts)')
#ax.set_ylim(0,50)

#ax.set_xscale('log')
ax.set_xlim(0,40)
ax.set_xlabel('energy (keV)')
"""
