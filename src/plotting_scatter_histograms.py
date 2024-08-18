# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:25:45 2024

@author: marko
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from os.path import join
import pandas as pd


current_path = os.getcwd()

#path to pickle file
pk_file = "pickle/data_clustered.pkl"

pk_path = join(current_path, pk_file)

#loading pickle
with open(pk_path, 'rb') as f:
    data = pickle.load(f)

# Convert the list of histograms into array
histograms = np.array(data['RGB_Histogram'])  # Initially, this might be an object array
histograms = np.vstack(histograms)  

#get relevant data
image_ids = np.array(data['Image_ID'])       
clusters = np.array(data['cluster_histogram'])

# Ensure the shapes are as expected
print(f"Histograms shape after reshaping: {histograms.shape}")  
print(f"Image IDs length: {len(image_ids)}")    
print(f"Clusters length: {len(clusters)}")     

# Count the number of unique clusters
unique_clusters = np.unique(clusters)
num_clusters = len(unique_clusters)
print(f"Number of clusters: {num_clusters}")
tsne = TSNE(n_components=2, random_state=42)
#early_exaggeration=25
embedding = tsne.fit_transform(histograms)

# Convert clusters to categorical type
clusters = pd.Categorical(clusters, categories=sorted(unique_clusters))

print("dimensionality reduction finished")
#%%
#plotting data
plt.figure(figsize=(12, 8))

scatter = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7, hue_order=sorted(unique_clusters))

#legend
handles, labels = scatter.get_legend_handles_labels()
scatter.legend(handles=handles, labels=[f"Cluster {int(label)}" for label in labels])

plt.title('TSNE Scatter Plot of RGB Histograms Colored by Clusters')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.legend(title=f'Clusters (Total: {num_clusters})')
plt.show()
