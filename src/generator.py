# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:25:48 2024

@author: marko
"""

"""nimmt als input den pfad zu den bildern un schreibt dann den namen, dimensionen und farbwerte(durchschnitt) raus, muss noch schauen, dass der weitermacht und nciht die csv neuschreibt
und durch folder iterieren kann"""

# Required Libraries 

import os
from os.path import join, isfile
from pathlib import Path 
import numpy 
import cv2 
import argparse 
import numpy 
import csv 
from tqdm import tqdm
import numpy as np


#ID for each image, refered to in csv-file
image_id = 0

csv_file = "csv\images.csv" #"C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
current_path = os.getcwd()
csv_path = join(current_path, csv_file)

def create_csv(csv_path):
    # Check whether the CSV 
    # exists or not if not then create one. 
        	
    #creating csv if not existing
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Name", "Height", "Width", "Channels", "Avg Blue", "Avg Red", "Avg Green"])
            
def image_generator(args):
    
    path = Path(args.folder)

    #creating a list with all paths already loaded into csv
    list_img = []
    current_ID = -1
    #C:\Users\marko\Documents\viertes_semester\BigData\Image_recommender_Big_Data\src\csv
    #C:/Users/marko/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/
    with open('csv/images.csv', mode ='r')as file:
      csvFile = csv.reader(file)
      
      for lines in csvFile:
          list_img.append(lines[1])
          current_ID += 1
          #print(current_ID)
          """if current_ID == 'ID':
              current_ID = 0"""
              
    gen_uptodate = False #variable we use to check if the generator is yielding new images or old ones
    
    # generator that runs image files from our given directory as the parameter
    for root, _, files in os.walk(path):
        
        for file in tqdm(files, total=len(files)):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                
                if gen_uptodate == False: #generator still yielding old images
                #checking if image is already in database
                    if image_path not in list_img:
                        gen_uptodate = True #set to True if one image has not been added to csv yet
                        #print(f"gen_uptodate: {gen_uptodate}")
                        
                if gen_uptodate == True: #if one is new, all folowing will be new too 
                    #print("new image loaded into csv")
                    #loading the image
                    img = cv2.imread(image_path) 
                    
                    
                    #print(image_path)
                    #print(current_ID)
                    image_id = current_ID
                    #print(img)
                    yield img, image_path, image_id
                    #setting ID counter up
                    current_ID += 1
                    #print(current_ID)

def image_generator_with_batch_MARIE(args.folder, args.batch_size):

    """to Marko:
    this is the image_generator function but with batch function.
    your latest changes are not included yet,
    maybe you can have a look at how to do this? I was not sure"""

    path = Path(args.folder)
    batch = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                global image_id
                image_id += 1
                batch.append((image_path, image_id))
                if len(batch) == args.batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch
                
def get_data(args, img, image_path, image_id, csv_path):
    """
    # Argparse function to get 
    # the path of the image directory 
    ap = argparse.ArgumentParser() 
    
    ap.add_argument("-f", "--folder", 
    				required = True, 
    				help = "Path to folder") 
    ap.add_argument("-b", "--batch_size", 
                    required= False,
                    help = "Number of images to load")
    
    args = vars(ap.parse_args()) """
    
        
    # Program to find the 
    # colors and embed in the CSV 
    #mypath, image_id = image_generator(args)
    image_path = str(image_path)
    #print(f"path:{image_path}")
    #print(type(image_path))
    batch_size = args.batch_size
    batch_size = int(batch_size)
    
    #loading the image
    #img = cv2.imread(image_path) 
    #print(img)
    h,w,c = img.shape
    
    """img_o = Image.open(image_path)
    form = img_o.format"""
       
    avg_color_per_row = numpy.average(img, axis = 0) 
    avg_color = numpy.average(avg_color_per_row, axis = 0) 
    #img_name = os.path.basename(image_path) #only img name, without whole path
   	
    """with open(csv_path, 'a', newline = '') as file: 
   		writer = csv.writer(file) 
   		writer.writerow([image_id, image_path, h, w, c, 
   						avg_color[0], avg_color[1], 
   						avg_color[2]]) 
    file.close() """
    return image_id, image_path, h, w, c, avg_color

def get_data_MARIE(image_path, image_id, csv_path):

    """same here: I'm not sure what you want to keep from the function above,
    but I figured it should work with this too.
    But I also only tried it with the rest of my code which
    was not quite the same"""

    # load the image
    img = cv2.imread(image_path)

    #get the shape and color data
    h, w, c = img.shape
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    
    #write the data into the csv
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_id, image_path, h, w, c, avg_color[0], avg_color[1], avg_color[2]])


"""
#method which combines the workflow of generating images and saving the wanted data into a csv
def generate(args):
    create_csv(args, csv_path)
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
            print("\nimage data loaded into csv") 
    except:
        StopIteration
        print("\nno new images to load into database")"""
     
    