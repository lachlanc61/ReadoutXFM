import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from sklearn import datasets, decomposition, manifold, preprocessing
from sklearn.cluster import KMeans
import umap.umap_ as umap

import config

#-----------------------------------
#CONSTANTS
#-----------------------------------
KCMAPS=["Accent","Set1"]

#-----------------------------------
#CLASSES
#-----------------------------------
"""
reducers = [
    (manifold.TSNE, {"perplexity": 50}),
    # (manifold.LocallyLinearEmbedding, {'n_neighbors':10, 'method':'hessian'}),
    (manifold.Isomap, {"n_neighbors": 30}),
    (manifold.MDS, {}),
    (decomposition.PCA, {}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
]
"""
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
#FUNCTIONS
#-----------------------------------

def reduce(data):

    embedding=np.zeros((nred,len(elements),2))
    clusttimes=np.zeros(nred)

    i = 0
    for reducer, args in reducers:
        redname=repr(reducers[i][0]()).split("(")[0]
        start_time = time.time()
        print(f'REDUCER {i+1} of {nred}: {redname} across {len(elements)} elements')

        if config.FORCERED:
            print("running", redname)
            embed = reducer(n_components=2, **args).fit_transform(data)
            np.savetxt(os.path.join(config.odir, redname + ".txt"), embed)
        else:
            print("loading from file", redname)
            embed = np.loadtxt(os.path.join(config.odir, redname + ".txt"))
        
        clusttimes[i] = time.time() - start_time
        embedding[i,:,:]=embed
        i += 1

    return embedding, clusttimes


def dokmeans(embedding):
    categories=np.zeros((nred,len(elements)))

    
    for i in np.arange(0,nred):
        redname=repr(reducers[i][0]()).split("(")[0]
        embed = embedding[i,:,:]

        print(f'KMEANS clustering {i+1} of {nred}, reducer {redname} across {len(elements)} elements')

        if config.FORCEKMEANS:
            print("running", redname)
            kmeans.fit(embed)
            categories[i]=kmeans.labels_
            np.savetxt(os.path.join(config.odir, redname + "_kmeans.txt"), categories[i])
        else:
            print("loading from file", redname)
            categories[i]=np.loadtxt(os.path.join(config.odir, redname + "_kmeans.txt"))
    return categories

def clustplt(embedding, categories, clusttimes):
   #set  figure params
    sns.set(context="paper", style="white")
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    )

    ax_list = []    #list of axes

    for i in np.arange(0,nred):

        redname=repr(reducers[i][0]()).split("(")[0]
        embed = embedding[i,:]
        elapsed_time = clusttimes[i]
        
        plotindex=i
        print("PLOTINDEX",plotindex)
        ax = plt.subplot(nred, 2, plotindex+1)
        print(redname, "colourmap", KCMAPS[i])
        print(redname, "categories", categories[i])
        ax.scatter(*embed.T, s=10, c=categories[i], cmap=KCMAPS[i], alpha=0.5)
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
        ax_list[plotindex].set_xlabel(redname, size=16)
        ax_list[plotindex].xaxis.set_label_position("top")

        print(ax_list)        


        plotindex=i+2

        print("PLOTINDEX",plotindex)
        ax = plt.subplot(nred, 2, (plotindex+1))
        catmap=np.reshape(categories[i], [-1,config.MAPX])
        ax.imshow(catmap, cmap=KCMAPS[i])


        ax_list.append(ax)
        ax_list[plotindex].set_xlabel(redname, size=16)
        ax_list[plotindex].xaxis.set_label_position("top")

        print(ax_list)

    plt.setp(ax_list, xticks=[], yticks=[])

    plt.tight_layout()

    plt.savefig(os.path.join(config.odir, 'clusters.png'), dpi=150)
    plt.show()
    return
    """
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.04, top=0.96, wspace=0.05, hspace=0.1
    )

    i=0
    for reducer, args in reducers:
        catmap=np.loadtxt(os.path.join(odir, redname + "_kgrid.txt"))
        
        ax = plt.subplot(n_cols, 1, i+1)
        ax.imshow(catmap, cmap=CMAPS[i])


        i+=1    

    plt.savefig(os.path.join(odir, 'catmaps.png'), dpi=150)
    plt.show()
    """

#-----------------------------------
#INITIALISE
#-----------------------------------

nred = len(reducers)
elements=np.arange(0,config.MAPX*config.MAPY)
