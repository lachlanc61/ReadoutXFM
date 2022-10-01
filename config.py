
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
DOCOLOURS=True
DOCLUST=True
SEPSUMSPEC=True
SAVEPXSPEC=True


DOBG=False
LOWBGADJUST=True



nclust=10

CMAP='Set1'
MAPX=256    #for geo2
MAPY=126

#debug flags
DEBUG=False     #debug flag (pixels)
DEBUG2=False    #second-level debug flag (channels)

#shortcut flags
SHORTRUN=False  #stop after first X% of pixels
shortpct=10    #% of lines

#recalc flags
FORCEPARSE=True
FORCERED=True        #always recalc dimensionality reduction
FORCEKMEANS=True     #always recalc kmeans 

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script

#infile = "geo_nfpy11.GeoPIXE"
#infile = "geo_ln_chle.GeoPIXE"
#infile = "geo_dwb12-2.GeoPIXE"
infile = "geo2.GeoPIXE"    #assign input file
outfile="pxspec"

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