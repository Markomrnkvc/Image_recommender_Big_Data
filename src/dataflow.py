# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 02:36:11 2024

@author: marko
"""
from generator import create_csv, image_generator, get_data, data_writer
from dataframe import create_pk, save_in_df
import generator


import os
from os.path import join, isfile
from pathlib import Path 
import numpy 
import cv2 
import argparse 
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
csv_file = "csv\images.csv" #"C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
error_file = "csv\error_images.csv"

#path to pickle file
pk_file = "pickle\data.pk"

csv_path = join(current_path, csv_file)
error_path = join(current_path, error_file)
pk_path = join(current_path, pk_file)

image_id = 0
#method which combines the workflow of generating images and saving the wanted data into a csv
def dataflow(args):
    create_csv(args, csv_path)
    create_pk(pk_path)
    
    try:
        gen = next(image_generator(args))
        if gen == None:
                print("\nNo new images")
                return
        
        #opening pickle
        df = pd.read_pickle(pk_path)
        
        #for img ,image_path, image_id in tqdm(image_generator(args), total=444880):
        for img ,image_path, image_id in image_generator(args):
            #print(image_id)
            #getting data out of images
            try:
                image_id, image_path, h, w, c, avg_color = get_data(args, img, image_path, image_id, csv_path)
                #writing data into csv
                data_writer(image_id, image_path, h, w, c, avg_color, csv_path)
                
                
                #saving data in pickle file
                embedding = random.randint(0,1000) #need placeholder, no embeddings yet
                
                save_in_df(embedding, image_id, h, w, c, avg_color, df)
                
                #closing pickle after 50 images to save progress
                if image_id % 50 == 0:
                    #closing pickle to save
                    df.to_pickle(pk_path)
                    #opening pickle
                    df = pd.read_pickle(pk_path)
            except:
                AttributeError
                print(f"\nError loading image {image_path}")
                
                with open(error_path, 'a', newline = '') as file: 
               		writer = csv.writer(file) 
               		writer.writerow([image_path]) 
                file.close() 
           
                
            #print("\nimage data loaded into csv") 
        print(f"number currently loaded images: {image_id}, {image_path}")
        #closing pickle at end of generator to save
        df.to_pickle(pk_path)

    except:
        StopIteration
        
        print(f"\nnumber currently loaded images: {image_id}, {image_path}")
        #closing pickle at end of generator to save
        df.to_pickle(pk_path)
        print("\nno new images to load into database or generator interrupted manually")

