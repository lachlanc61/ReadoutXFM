import numpy as np
import pybaselines.smooth

#-----------------------------------
#MODIFIABLE CONSTANTS
#-----------------------------------

SNIPWINDOW=50   #width-window for SNIP algorithm - 50 is default
LOWCUT=80       #low cut point for SNIP

#-----------------------------------
#FUNCTIONS
#-----------------------------------

def fitbaseline(y,noisy):
    bg = pybaselines.smooth.snip(y[LOWCUT:],SNIPWINDOW)[0]
    bg=np.pad(bg, (LOWCUT, 0), 'constant', constant_values=(bg[0], 0))
    if noisy:
        bg=3*bg+1
    y=y-bg
    y[y<0] = 0
    return y, bg