import pandas as pd
import numpy as np
import random
import time

npx=5000
nchan=4096

#CLASSES
class Map:
    def __init__(self, n):
        #header values
        self.numpx = n         

class PixelSeries:
    def __init__(self, map):
        #initialise pixel value arrays
        self.pxlen=np.zeros(map.numpx,dtype=np.uint16)
        self.xidx=np.zeros(map.numpx,dtype=np.uint16)
        self.yidx=np.zeros(map.numpx,dtype=np.uint16)
        self.det=np.zeros(map.numpx,dtype=np.uint16)
        self.dt=np.zeros(map.numpx,dtype=np.uint16)

map = Map(npx)

starttime = time.time()             #init timer

doset=1
#dopandas=True

if doset == 0:  #using pandas
    
    df = pd.DataFrame()
    #df = df.assign(pxlen=np.random.rand(npx))
    #df = df.assign(xidx=np.random.rand(npx))

    df = df.assign(pxlen=np.zeros(npx,dtype=np.uint16))
    df = df.assign(xidx=np.zeros(npx,dtype=np.uint16))
    df = df.assign(yidx=np.zeros(npx,dtype=np.uint16))
    df = df.assign(det=np.zeros(npx,dtype=np.uint16))
    df = df.assign(dt=np.zeros(npx,dtype=np.uint16))

    for i in np.arange(npx):
        df.at[i,'pxlen'] = i+4
        df.at[i,'xidx'] = random.randint(0,4096)
        df.at[i,'yidx'] = random.randint(0,4096)
        df.at[i,'det'] = random.randint(0,4096)
        df.at[i,'dt'] = random.randint(0,4096)    

 
    print(df.iloc[npx-1]['pxlen'])
    print("PANDAS")

elif doset == 1:    #using numpy arrays
    pxlen=np.zeros(npx,dtype=np.uint16)
    xidx=np.zeros(npx,dtype=np.uint16)
    yidx=np.zeros(npx,dtype=np.uint16)
    det=np.zeros(npx,dtype=np.uint16)
    dt=np.zeros(npx,dtype=np.uint16)

    for i in np.arange(npx):
        pxlen[i] = i+4
        xidx[i] = random.randint(0,4096)
        yidx[i] = random.randint(0,4096)
        det[i] = random.randint(0,4096)
        dt[i] = random.randint(0,4096)        

    
    print(pxlen[npx-1],xidx[npx-1])
    print("NUMPY")

elif doset == 3:    #using numpy arrays via a class

    pxseries=PixelSeries(map)

    for i in np.arange(npx):
        pxseries.pxlen[i] = i+4
        pxseries.xidx[i] = random.randint(0,4096)
        pxseries.yidx[i] = random.randint(0,4096)
        pxseries.det[i] = random.randint(0,4096)
        pxseries.dt[i] = random.randint(0,4096)  

    
    print(pxseries.pxlen[npx-1],pxseries.xidx[npx-1])
    print("NP CLASS")

runtime = time.time() - starttime

print("TIME (s):", runtime)



"""
times (n=5000):  
    #method             #time (s)

    PANDAS              0.272   (!!!)
    NUMPY               0.0166
    NP CLASS            0.0172

pandas is sloooow.... use a custom class
"""