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

def fit_cluster(n_clusters = 10):
    print("Clustering the collected data")
    
    #loading pickle file with data
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
    
    cols_to_remove_emb = [ 'Image_ID', 'Height', 'Width', 'Channels',
                      'Avg Blue', 'Avg Red', 'Avg Green', 'RGB_Histogram', 'Perceptual_Hash']
    
    #data_kmeans_hash = data.drop(cols_to_remove_hash, axis = 1)
    
    #getting data for histograms clustering
    histo_data = data.drop(cols_to_remove_rgb, axis = 1)
    #print(histo_data.iloc[0]['RGB_Histogram'])
    histograms_col = pd.DataFrame(data['RGB_Histogram'].tolist())
    
    
    #getting data for perceptual hash clustering
    phash_data = data.drop(cols_to_remove_hash, axis = 1)
    phash_col = pd.DataFrame(data['Perceptual_Hash'].tolist())
    
    #getting data for embedding clusteting
    emb_data = data.drop(cols_to_remove_emb, axis = 1)
    emb_col = pd.DataFrame(data['Embeddings'].tolist())
    #print(data)
    
    #hash_df = pd.DataFrame(data_kmeans_hash['Perceptual_Hash'].tolist())
    
    
    #scaling of data
    scaler = StandardScaler()
    
    scaler.fit(histograms_col)
    histograms_col = pd.DataFrame(scaler.transform(histograms_col),
                               columns= histograms_col.columns )
    
    scaler.fit(phash_col)
    histograms_col = pd.DataFrame(scaler.transform(phash_col),
                               columns= phash_col.columns )
    
    scaler.fit(emb_col)
    histograms_col = pd.DataFrame(scaler.transform(emb_col),
                               columns= emb_col.columns )
    
    #print(data_kmeans)
    #print(data_kmeans.isna().any()) #--> keine phashes fehlen
    
    #data_kmeans['Perceptual_Hash'] = data_kmeans['Perceptual_Hash'].apply(lambda x: ','.join(map(str, x)))
    
    #initializing clustering model
    def initializing_kmeans(n_clusters):
        print(f"\ninitializing kmeans-models with {n_clusters} clusters")
        
        kmeans_histo = KMeans( n_clusters = n_clusters, random_state = 0)
        
        kmeans_phash = KMeans( n_clusters = n_clusters, random_state = 0)
        
        kmeans_emb = KMeans( n_clusters = n_clusters, random_state = 0)
        
        return kmeans_emb, kmeans_histo, kmeans_phash
    #fitting model to methods
    def fit_and_save_kmeans(n_clusters):
        print("\nfitting and saving kmeans-models\n")
        kmeans_emb, kmeans_histo, kmeans_phash = initializing_kmeans(n_clusters)
        
        #fitting kmeans models
        kmeans_histo.fit(histograms_col)
        kmeans_phash.fit(phash_col)
        kmeans_emb.fit(emb_col)
    #kmeans.cluster_centers_
    
        
        modelfile_histo = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_histograms.pkl"
        modelfile_phash = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_perceptualhashes.pkl"
        modelfile_emb = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_embeddings.pkl"
        
        #speichern der modelle
        with open(modelfile_histo, 'wb') as modelfile_histo:
            pickle.dump(kmeans_histo, modelfile_histo)
            
        with open(modelfile_phash, 'wb') as modelfile_phash:
            pickle.dump(kmeans_phash, modelfile_phash)
            
        with open(modelfile_emb, 'wb') as modelfile_emb:
            pickle.dump(kmeans_emb, modelfile_emb)
    
        
        #adding clusters to data.pk
        data['cluster_histogram'] = kmeans_histo.labels_
        data['cluster_embedding'] = kmeans_emb.labels_
        data['cluster_phash'] = kmeans_phash.labels_
        
        #saving file
        #print(data.cluster_histogram.value_counts())
        
        data.to_pickle(file)
    
    """with open(filename, "wb") as f:
            pickle.dump({"Image_ID": data_kmeans_rgb.Image_ID, "model": kmeans.labels_}, f)
    """
    #histo_data.to_pickle(filename)
    
    
    fit_and_save_kmeans(n_clusters)
    
fit_cluster()
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
modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_histograms.pkl"

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
    
    print([histogram])
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
#modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"

def fit_cluster(img, img_path, method, modelfile):#, histogram, embedding, phash):
    if method == 'histogram':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_histograms.pkl"
       
    elif method == 'embeddings':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_embeddings.pkl"
        
    elif method == 'hashes':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_perceptualhashes.pkl"
        
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
            'RGB_Histogram': histogram
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data['RGB_Histogram'].tolist())
        
        
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        
    elif method == 'embeddings':
        new_data = pd.DataFrame({
            'Name': img_path,
            'Embeddings': [embedding]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data['Embeddings'].tolist())
        
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
    elif method == 'hashes':
        new_data = pd.DataFrame({
            'Name': img_path,
            'Perceptual_Hash': [phash_vector]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data['Perceptual_Hash'].tolist())
        
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        
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
    new_data['cluster'] = kmeans.predict(new_data_df)


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