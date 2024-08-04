# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 02:36:11 2024

@author: marko
"""
from generator import create_csv, image_generator, get_data, data_writer
from dataframe import create_pk, save_in_df


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

#ID for each image, refered to in csv-file
#image_id = 0

#getting current path
current_path = os.getcwd()

#path to csv-file
csv_file = "csv\images.csv" #"C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
#path to pickle file
pk_file = "pickle\data.pk"

csv_path = join(current_path, csv_file)
pk_path = join(current_path, pk_file)

#method which combines the workflow of generating images and saving the wanted data into a csv
def dataflow(args):
    create_csv(args, csv_path)
    create_pk(pk_path)
    
    try:
        gen = next(image_generator(args))
        if gen == None:
                print("\nNo new images")
                return
        #print(next(image_generator(args)))
        
        for img ,image_path, image_id in image_generator(args):
            print(image_id)
            #getting data out of images
            image_id, image_path, h, w, c, avg_color = get_data(args, img, image_path, image_id, csv_path)
            #writing data into csv
            data_writer(image_id, image_path, h, w, c, avg_color, csv_path)
            
            
            #saving data in pickle file
            embedding = random.randint(0,1000) #need placeholder, no embeddings yet
            save_in_df(embedding, image_id, h, w, c, avg_color, pk_path)
            print("\nimage data loaded into csv") 
    except:
        StopIteration
        print("\nno new images to load into database")

