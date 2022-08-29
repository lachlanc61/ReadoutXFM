import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
MAPX=256    #for geo2
MAPY=126

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
#infile = "leaf2_overview.GeoPIXE"    #assign input file
infile = "geo2.GeoPIXE"    #assign input file

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


fig=plt.figure()
axr=fig.add_subplot(311)
axg=fig.add_subplot(312)
axb=fig.add_subplot(313)

#-----------------------------------
#MAIN START
#-----------------------------------
rvals=np.loadtxt(os.path.join(odir, "rvals.txt"))
gvals=np.loadtxt(os.path.join(odir, "gvals.txt"))
bvals=np.loadtxt(os.path.join(odir, "bvals.txt"))

print("RED",rvals)
print("GREEN",gvals)
print("BLUE",bvals)


rreshape=np.reshape(rvals, (-1, MAPX))
greshape=np.reshape(rvals, (-1, MAPX))
breshape=np.reshape(rvals, (-1, MAPX))


diff=np.subtract(rvals,gvals)
diff=np.reshape(diff, (-1, MAPX))
axr.imshow(rreshape)
axg.imshow(greshape)
axb.imshow(breshape)

print("redvals",rreshape.shape,rreshape[40,230])
print("bluevals",breshape.shape,breshape[40,230])
plt.show()

exit()

rgbarray = np.zeros((MAPY,MAPX,3), 'uint8')
rgbarray[..., 0] = rreshape*256
rgbarray[..., 1] = greshape*256
rgbarray[..., 2] = breshape*256

#    np.savetxt(os.path.join(odir, "rgb.txt"), rgbarray)
print(rgbarray.shape)
plt.imshow(rgbarray)

plt.show()
