# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:45:53 2024

@author: marko
"""
"""
wählen kmeans, da so alle Bilde reinem cluster zugeordnet werden,
 egal wie gut sie in dieses reinpassen (also werden immer dem zugeordnet was noch am besten passt)
 """
from histograms import hist
from phashes import perceptual_hashes
import cv2
import os
from os.path import join
import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  KMeans

def fit_cluster(n_clusters = 10):
    '''
    Returns three kmeans models based on 1. RGB histograms, 2. Resnet embeddings, 3. Perceptual Hashes
    
    Adds columns to data.pk ('cluster_histogram''cluster_embedding''cluster_phash') which include the cluster each image gets sorted to

            Parameters:
                    n_clusters (int): number of clusters; default; n_clusters = 10
            Returns:
                    data.pk (file): pickle file with all of the image data including clusters
                    
                    kmeans_model_histograms.pkl (file): includes kmeans model
                    
                    kmeans_model_embeddings.pkl (file): includes kmeans model
                    
                    kmeans_model_perceptualhashes.pkl (file): includes kmeans model
                    
            use-example: fit_cluster() 
                    
                    
    '''
    print("Clustering the collected data")
    
    #getting pickle file 
    current_path = os.getcwd()
    #path to pickle file
    pk_file = "pickle/data.pkl"
    pk_path = join(current_path, pk_file)
    #loading pickle file with data
    file = pk_path#"C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/data.pkl"
    
    #loading pickle file
    data = pd.read_pickle(file)
    """
    with open(file, 'rb') as file:
        data = pickle.load(file)
    
    data = pd.DataFrame(data)
    """
    """
    #entfernen der für kmeans unnötigen Featuren
    cols_to_remove_hash = ['Image_ID', 'Embeddings', 'Height', 'Width', 'Channels',
                      'Avg Blue', 'Avg Red', 'Avg Green', 'RGB_Histogram']
    
    cols_to_remove_rgb = [ 'Embeddings', 'Height', 'Width', 'Channels',
                      'Avg Blue', 'Avg Red', 'Avg Green', 'Perceptual_Hash']
    
    cols_to_remove_emb = [ 'Image_ID', 'Height', 'Width', 'Channels',
                      'Avg Blue', 'Avg Red', 'Avg Green', 'RGB_Histogram', 'Perceptual_Hash']
    """
    #data_kmeans_hash = data.drop(cols_to_remove_hash, axis = 1)
    
    #getting data for histograms clustering
    #histo_data = data.drop(cols_to_remove_rgb, axis = 1)
    #print(histo_data.iloc[0]['RGB_Histogram'])
    histograms_col = pd.DataFrame(data['RGB_Histogram'].tolist())
    
    
    #getting data for perceptual hash clustering
    #phash_data = data.drop(cols_to_remove_hash, axis = 1)
    phash_col = pd.DataFrame(data['Perceptual_Hash'].tolist())
    
    #getting data for embedding clusteting
    #emb_data = data.drop(cols_to_remove_emb, axis = 1)
    emb_col = pd.DataFrame(data['Embeddings'].tolist())
    
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
    
        
        modelfile_histo = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_histograms.pkl"
        modelfile_phash = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_perceptualhashes.pkl"
        modelfile_emb = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_embeddings.pkl"
        
        #saving the models
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
        data.to_pickle(file)
    
    
    fit_and_save_kmeans(n_clusters)
    
    print("clustering finished!")
    




"""
einladen eines bildes noch drin gelassen, sollten wir später rausnehmen
"""
"""
image_path = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg"

img = cv2.imread(image_path)
#modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"
#img = cv2.imread(image_path) 

histogram = hist(img)

embedding = "a"

phash_vector = perceptual_hashes(img)
"""
def predict_cluster(img, img_path, args, data):#, histogram, embedding, phash):
    '''
    Returns the cluster (int) the uploaded image belongs to
    
    
            Parameters:
                    n_clusters (int): number of clusters
                    img: loaded image
                    img_path: oath of the image
                    method: method of comparison, choose between  ['histogram', 'embeddings', 'hashes']
                    data: data we need for comparing the image, based on which method we choose (histogram of the image, embedding of the image, perceptual hash of the image)
            Returns:
                    cluster (int): number of the cluster the image belongs to
                    
            use-example: cluster = predict_cluster(img, image_path, method = 'histogram', data = histogram)
                    
    '''
    
    if args.method == 'histogram':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_histograms.pkl"
        
        #data = histogram
       
    elif args.method == 'embeddings':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_embeddings.pkl"
        
        #data = embedding
        
    elif args.method == 'hashes':
        modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_models/kmeans_model_perceptualhashes.pkl"
        
        #data = phash_vector
        
    with open(modelfile, 'rb') as file:
        kmeans = pickle.load(file)


    
    if args.method == 'histogram':
        new_data = pd.DataFrame({
            'Name': img_path,
            'RGB_Histogram': data
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data['RGB_Histogram'].tolist())
        
        
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        
    elif args.method == 'embeddings':
        new_data = pd.DataFrame({
            'Name': img_path,
            'Embeddings': [data]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data['Embeddings'].tolist())
        
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
    elif args.method == 'hashes':
        new_data = pd.DataFrame({
            'Name': img_path,
            'Perceptual_Hash': [data]
        })
        
        # Konvertiere die neue Liste der perceptual hashes in ein DataFrame
        new_data_df = pd.DataFrame(new_data['Perceptual_Hash'].tolist())
        
        #scaling data
        scaler = StandardScaler()

        scaler.fit(new_data_df)
        new_data_df = pd.DataFrame(scaler.transform(new_data_df),
                                   columns= new_data_df.columns )
        
    # Weise den neuen Daten Cluster zu, ohne das Modell neu zu trainieren
    new_data['cluster'] = kmeans.predict(new_data_df)


    return(new_data.iloc[0]['cluster'])
    
