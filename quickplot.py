
import numpy as np
import config
import src.clustering as clustering

#assign dummy data - module will read from file
data=np.zeros(200)

if config.DOCLUST:
    embedding, clusttimes = clustering.reduce(data)
    categories = clustering.dokmeans(embedding)
    print("categories full")
    print(categories)
    clustering.clustplt(embedding, categories, clusttimes)