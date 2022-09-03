import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn import decomposition
from sklearn.cluster import KMeans
import umap.umap_ as umap

import config

#-----------------------------------
#CONSTANTS
#-----------------------------------
KCMAPS=["Accent","Set1"]    #colourmaps for kmeans

#-----------------------------------
#CLASSES
#-----------------------------------
"""
#full reducer list here
from sklearn import datasets, decomposition, manifold, preprocessing

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
    """
    receives arrays from reducers and kmeans
    + time to cluster

    plots Nx2 plot for each reducer

    https://towardsdatascience.com/clearing-the-confusion-once-and-for-all-fig-ax-plt-subplots-b122bb7783ca
    """    
        
    #create figure and ax matrix
    #   gridspec adjusts widths of subplots in each row
    fig, (ax) = plt.subplots(nred, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.tight_layout(pad=2)
 
    #fig.subplots_adjust(
    #    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    #)

    #for each reducer
    for i in np.arange(0,nred):
        #get the reducer's name
        redname=repr(reducers[i][0]()).split("(")[0]
        #read in the embedding xy array and time
        embed = embedding[i,:]
        elapsed_time = clusttimes[i]
        
        #assign index in plot matrix
        plotid=(i,0)

        #adjust plotting options
        ax[plotid].set_xlabel(redname, size=16)
        ax[plotid].xaxis.set_label_position("top")

        #create the scatterplot for this reducer
        ax[plotid].scatter(*embed.T, s=10, c=categories[i], cmap=KCMAPS[i], alpha=0.5)

        #add the runtime as text
        ax[plotid].text(
            0.99,
            0.01,
            "{:.2f} s".format(elapsed_time),
            transform=ax[plotid].transAxes,
            size=14,
            horizontalalignment="right",
        )
        
        #assign index for category map for this reducer
        plotid=(i,1)

        #reshape the category list using the image dimension globals
        catmap=np.reshape(categories[i], [-1,config.MAPX])

        #show this category image
        ax[plotid].imshow(catmap, cmap=KCMAPS[i])

    #initalise the final plot, clear the axes
    plt.setp(ax, xticks=[], yticks=[])

    #save and show
    plt.savefig(os.path.join(config.odir, 'clusters.png'), dpi=150)
    plt.show()
    return

#-----------------------------------
#INITIALISE
#-----------------------------------

nred = len(reducers)
elements=np.arange(0,config.MAPX*config.MAPY)

