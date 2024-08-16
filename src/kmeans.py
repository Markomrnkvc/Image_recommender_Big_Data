# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:45:53 2024

@author: marko
"""
"""
wählen kmeans, da so alle Bilde reinem cluster zugeordnet werden,
 egal wie gut sie in dieses reinpassen (also werden immer dem zugeordnet was noch am besten passt)
 """
import argparse
import pickle
import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  KMeans
import time 

start = time.time()

file = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/data.pk"

filename = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/rgb_cluster.pk"
"""with open(file, 'rb') as f:
    timeseries = pickle.loads(f.read())
    #print(timeseries)"""
   
data = pd.read_pickle(file)
 

#entfernen der für kmeans unnötigen Featuren
cols_to_remove_hash = ['Image_ID', 'Embeddings', 'Height', 'Width', 'Channels',
                  'Avg Blue', 'Avg Red', 'Avg Green', 'RGB_Histogram']

cols_to_remove_rgb = [ 'Embeddings', 'Height', 'Width', 'Channels',
                  'Avg Blue', 'Avg Red', 'Avg Green', 'Perceptual_Hash']
#data_kmeans_hash = data.drop(cols_to_remove_hash, axis = 1)
histo_data = data.drop(cols_to_remove_rgb, axis = 1)

#print(data)

#hash_df = pd.DataFrame(data_kmeans_hash['Perceptual_Hash'].tolist())
RGB_df = pd.DataFrame(data['RGB_Histogram'].tolist())
"""
scaler = StandardScaler()
scaler.fit(RGB_df)
RGB_df = pd.DataFrame(scaler.transform(RGB_df),
                           columns= RGB_df.columns )"""
#print(data_kmeans)
#print(data_kmeans.isna().any()) #--> keine phashes fehlen

#data_kmeans['Perceptual_Hash'] = data_kmeans['Perceptual_Hash'].apply(lambda x: ','.join(map(str, x)))

kmeans = KMeans( n_clusters = 20, random_state = 0)
#kmeans.fit(hash_df)
kmeans.fit(RGB_df)
#kmeans.cluster_centers_


modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"
#speichern des modells
with open(modelfile, 'wb') as modelfile:
    pickle.dump(kmeans, modelfile)
    
    
histo_data['cluster'] = kmeans.labels_#


"""with open(filename, "wb") as f:
        pickle.dump({"Image_ID": data_kmeans_rgb.Image_ID, "model": kmeans.labels_}, f)
"""
histo_data.to_pickle(filename)
#print(data.cluster.value_counts())

end = time.time()
print(f" runtime: {end - start}")




#%%

import random
from histograms import hist
import cv2
import os
import time 

""" method for predict cluster for an input image, based on the saved kmeans model in /pickle, 
output is an integer number (number of cluster)
"""

start = time.time()

image_path = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg"
modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"

def predict_cluster():
    with open(modelfile, 'rb') as file:
        kmeans = pickle.load(file)

    img = cv2.imread(image_path) 
    histogram = hist(img)
    #print(len(histogram))
    #print(histogram)
    # Beispiel: Neue Daten
    new_data = pd.DataFrame({
        'Image_ID': 1237812,
        'RGB_Histogram': [histogram]
    })
    
    # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
    new_hash_df = pd.DataFrame(new_data['RGB_Histogram'].tolist())
    
    #print(new_hash_df)
    #print(type(kmeans))
    # Weise den neuen Daten Cluster zu, ohne das Modell neu zu trainieren
    new_data['cluster'] = kmeans.predict(new_hash_df)

    print(new_data.cluster)
    
predict_cluster()

end = time.time()
print(f" runtime: {end - start}")


"""
    so wird ein neues Bild einem Cluster hinzugefügt!!!!!!!!!!
"""
#%%



import random
from histograms import hist
from phashes import perceptual_hashes
import cv2
import os
import time 

start = time.time()


image_path = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg"

img = cv2.imread(image_path)
modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"

def fit_cluster(img, img_path, method, modelfile):#, histogram, embedding, phash):
    if method == 'histogram':
        #modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model_histogram.pkl"
        modelfile = modelfile
    elif method == 'embeddings':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model_embedding.pkl"
        
    elif method == 'hashes':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model_hashes.pkl"
    

    with open(modelfile, 'rb') as file:
        kmeans = pickle.load(file)

    #img = cv2.imread(image_path) 
    
    histogram = hist(img)
    
    embedding = "a"
    
    phash_vector = perceptual_hashes(img)
    
    #print(len(histogram))
    #print(histogram)
    # Beispiel: Neue Daten
    
    if method == 'histogram':
        new_data = pd.DataFrame({
            'Name': img_path,
            'RGB_Histogram': [histogram]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_hash_df = pd.DataFrame(new_data['RGB_Histogram'].tolist())
        
    elif method == 'embeddings':
        new_data = pd.DataFrame({
            'Name': img_path,
            'Embeddings': [embedding]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_hash_df = pd.DataFrame(new_data['Embeddings'].tolist())
        
    elif method == 'hashes':
        new_data = pd.DataFrame({
            'Name': img_path,
            'Perceptual_Hash': [phash_vector]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_hash_df = pd.DataFrame(new_data['Perceptual_Hash'].tolist())
        
    """ hier muss das einladen eines neuen bildes hin, am besten neue file mit kompletter funktionen (wie dataflow)
    
    new_data = pd.DataFrame({
        'Image_ID': 1237812,
        'RGB_Histogram': [histogram]
    })
    
    # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
    new_hash_df = pd.DataFrame(new_data['RGB_Histogram'].tolist())
    
    #print(new_hash_df)
    #print(type(kmeans))
    """
    
    # Weise den neuen Daten Cluster zu, ohne das Modell neu zu trainieren
    new_data['cluster'] = kmeans.predict(new_hash_df)


    return(new_data.iloc[0]['cluster'])
    
cluster = fit_cluster(img, image_path, method = 'histogram', modelfile=modelfile)
print(cluster)

end = time.time()
print(f" runtime: {end - start}")


"""
    so wird ein neues Bild einem Cluster hinzugefügt!!!!!!!!!!
"""
#%%

data = pd.read_pickle("C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/rgb_cluster.pk")
print(data.cluster.value_counts())