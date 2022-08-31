import os
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import time

from sklearn import datasets, decomposition, manifold, preprocessing
from sklearn.cluster import KMeans
from colorsys import hsv_to_rgb

import umap.umap_ as umap

#-----------------------------------
#USER MODIFIABLE VARIABLES
#-----------------------------------
MAPX=256    #for geo2
MAPY=126
NCHAN=4096
MAPX=256    #for geo2
MAPY=126

#workdir and inputfile
wdirname='data'     #working directory relative to script
odirname='out'      #output directory relative to script
#infile = "leaf2_overview.GeoPIXE"    #assign input file
infile = "geo2.GeoPIXE"    #assign input file


reducers = [
    (decomposition.PCA, {}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
]


kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=300,
    random_state=42
 )

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
 
sns.set(context="paper", style="white")

n_cols = len(reducers)
ax_index = 1
ax_list = []

# plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.figure(figsize=(10, 8))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)
#for data, labels in test_data:
#    print("cycle start",ax_index)

i=0

elements=np.arange(0,MAPX*MAPY)

for reducer, args in reducers:

    redname=repr(reducers[i][0]()).split("(")[0]

    start_time = time.time()

    embedding=np.loadtxt(os.path.join(odir, redname + ".txt"))

    kmeans.fit(embedding)

    elapsed_time = time.time() - start_time

    print(kmeans.labels_)

    ax = plt.subplot(1, n_cols, i+1)
    ax.scatter(*embedding.T, s=10, c=kmeans.labels_, cmap="Spectral", alpha=0.5)

    #else:
    #    ax.scatter(*embedding.T, s=10, c="red", cmap="Spectral", alpha=0.5)

    ax.text(
        0.99,
        0.01,
        "{:.2f} s".format(elapsed_time),
        transform=ax.transAxes,
        size=14,
        horizontalalignment="right",
    )
    
    ax_list.append(ax)

    print("reducer",redname, reducer)
    ax_list[i].set_xlabel(redname, size=16)
    ax_list[i].xaxis.set_label_position("top")

    i += 1

plt.setp(ax_list, xticks=[], yticks=[])

plt.tight_layout()
plt.show()
