# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:25:48 2024

@author: marko
"""

"""nimmt als input den pfad zu den bildern un schreibt dann den namen, dimensionen und farbwerte(durchschnitt) raus, muss noch schauen, dass der weitermacht und nciht die csv neuschreibt
und durch folder iterieren kann"""

# Required Libraries 

import os
from os.path import join, exists, isfile
from pathlib import Path 
import cv2
import argparse 
import csv 
from tqdm import tqdm
import numpy as np

#ID for each image, refered to in csv-file
image_id = 0

#path to csv-file
csv_file = "csv/images.csv" #"C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
error_file = "csv/error_images.csv"
current_path = os.getcwd()
csv_path = join(current_path, csv_file)
error_path = join(current_path, error_file)

def create_csv(args, csv_path):
    # Check whether the CSV exists or not if not then create one. 

    #creating csv if not existing
    if os.path.exists(csv_path) == False: 
    	with open(csv_path, 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(["Image_ID", "Name"])
    
    #creating csv if not existing
    if os.path.exists(error_path) == False:
        with open(error_path, 'w', newline = '') as file:
              writer = csv.writer(file)
              writer.writerow(["Name"])
            

def image_generator(args):
    path = Path(args.folder)
    list_img = []
    error_list_img = []
    current_ID = -1
    
    with open(csv_path, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            list_img.append(lines[1])
            current_ID += 1
    
    with open('csv/error_images.csv', mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            error_list_img.append(lines[0])
    
    for img_path in path.rglob('*.*'):
        print(f"Überprüfe Bild: {img_path}")
        if str(img_path) in list_img:
            print(f"{img_path} bereits verarbeitet, wird übersprungen.")
            continue
        if str(img_path) in error_list_img:
            print(f"{img_path} ist in der Fehlerliste, wird übersprungen.")
            continue
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Bild konnte nicht geladen werden: {img_path}")
                with open(error_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([str(img_path)])
                continue
            
            current_ID += 1
            yield img, str(img_path), current_ID
            print(f"Bild verarbeitet: {img_path} mit ID {current_ID}")

        except Exception as e:
            print(f"Fehler beim Verarbeiten von {img_path}: {e}")
            with open(error_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([str(img_path)])


def data_writer(image_id, image_path, csv_path):

    with open(csv_path, 'a', newline = '') as file: 
           writer = csv.writer(file) 
           writer.writerow([image_id, image_path]) 
    file.close()
