import os
import cv2

# fetching a random png image from my home directory, which has size 258 x 384
img_file = os.path.expanduser("/home/lachlan/CODEBASE/ColourspecXFM/data/190823_zba_full_CaMnK.png")

# read this image in as a NumPy array, using imread from scipy.misc
M = cv2.imread(img_file)
print(M)

print(M.shape)       # imread imports as RGBA, so the last dimension is the alpha channel

# now display the image from the raw NumPy array:
from matplotlib import pyplot as PLT

PLT.imshow(M)
PLT.show()