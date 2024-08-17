# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 02:36:11 2024

@author: marko
"""
from generator import create_csv, image_generator, data_writer
from dataframe import create_pk, save_in_df
from histograms import hist
from phashes import perceptual_hashes
from Resnet_Extraction import ResNet_Feature_Extractor


import os
from os.path import join, isfile
from pathlib import Path 
import numpy 
import csv 
from tqdm import tqdm
import random
import pandas as pd

#ID for each image, refered to in csv-file
#image_id = 0

#getting current path
current_path = os.getcwd()

#path to csv-file
csv_file = "csv/images.csv" #"C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
error_file = "csv/error_images.csv"

#path to pickle file
pk_file = "pickle/data.pk"

csv_path = join(current_path, csv_file)
error_path = join(current_path, error_file)
pk_path = join(current_path, pk_file)

image_id = 0
#method which combines the workflow of generating images and saving the wanted data into a csv
def dataflow(args):
    create_csv(args, csv_path)
    create_pk(pk_path)
    
    try:
        # Entferne diese Zeile, um den Generator nicht vorzeitig zu erschöpfen
        # gen = next(image_generator(args))
        # if gen == None:
        #     print("\nNo new images")
        #     return
        
        # Öffne das Pickle
        df = pd.read_pickle(pk_path)
        
        # Hauptverarbeitungsschleife
        for img, image_path, image_id in image_generator(args):
            print(f"Verarbeite Bild: {image_path} mit ID {image_id}")

            # Berechne Histogramm
            histogram = hist(img)
            
            # Berechne perceptual hashes
            phash_vector = perceptual_hashes(img)

            # Berechne Vektoren aus ResNet
            extractor = ResNet_Feature_Extractor(model_weights="imagenet")
            print("ResNet_Feature_Extractor erfolgreich initialisiert")
            resnet_embedding = extractor.extract_features(img)
            print(f"ResNet-Embedding erfolgreich berechnet für Bild ID {image_id}")
            
            # Schreibe Daten in CSV
            try:
                data_writer(image_id, image_path, csv_path)
                print(f"Bilddaten erfolgreich in CSV geschrieben: ID={image_id}, Pfad={image_path}")
            except Exception as e:
                print(f"Fehler beim Schreiben in die CSV-Datei: {e}")
            
            # Speichere Daten in der Pickle-Datei
            print("vor dem speichern in DF")
            save_in_df(resnet_embedding, image_id, histogram, phash_vector, df, pk_path)
            print("++++++ ++ + + + + NACH dem speichern in DF")
            print(df.head())
            
            # Schließe Pickle nach jeder 50. Iteration
            if image_id % 50 == 0:
                pd.to_pickle(df, pk_path)
                print(f"WWWWWWWWWWWWWWWWW WWW WWW WWW WWW WWW WWW WW THIS is THE dataFRAME after TO_piCKLE: {df}")
                df = pd.read_pickle(pk_path)
        
        print(f"number of currently loaded images: {image_id}, {image_path}")
        pd.to_pickle(df, pk_path)

    except StopIteration:
        print(f"\nno new images to load into database or generator interrupted manually")
        pd.to_pickle(df, pk_path)
    """except:
        StopIteration
        
        print(f"\nnumber currently loaded images: ") #{image_id}, {image_path}
        #closing pickle at end of generator to save
        df.to_pickle(pk_path)
        print("\nno new images to load into database or generator interrupted manually")"""

