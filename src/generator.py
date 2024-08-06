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
#from Pil import Image
#from main import args

#path of csv file for saving metadata
#my_file = args.folder #Path("C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/image_recomender/csv/images.csv") 

#ID for each image, refered to in csv-file
image_id = 0
#path to csv-file
csv_file = "csv\images.csv" #"C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
current_path = os.getcwd()
csv_path = join(current_path, csv_file)

def create_csv(args, csv_path):
    # Check whether the CSV 
    # exists or not if not then create one. 
    #my_images = args.folder #Path("C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/image_recomender/csv/images.csv") 

    #opening csv if existing, writing headers
    """if os.path.exists(csv_path):
            
        	f = open(csv_path, "w+") 
        	with open(csv_path, 'a', newline='') as file: 
        		
        		writer.writerow(["ID", "Name", "Height", 
        						"Width", "Channels", 
        						"Avg Blue", "Avg Red", 
        						"Avg Green"]) 
        	f.close() 
        	return 	writer = csv.writer(file) """
        	
    
    #creating csv if not existing
    if os.path.exists(csv_path) == False: 
    	with open(csv_path, 'w', newline = '') as file: 
    		writer = csv.writer(file) 
    		
    		writer.writerow(["ID", "Name", "Height", 
    						"Width", "Channels", 
    						"Avg Blue", "Avg Red", 
    						"Avg Green"]) 
            
def image_generator(args):#, path = Path("C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/images")):
    
    #if args.folder == True:
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
        
        #for file in tqdm(files, total=len(files)):
        for file in files:
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
   
def data_writer(image_id, image_path, h, w, c, avg_color, csv_path):
    
    with open(csv_path, 'a', newline = '') as file: 
   		writer = csv.writer(file) 
   		writer.writerow([image_id, image_path, h, w, c, 
   						avg_color[0], avg_color[1], 
   						avg_color[2]]) 
    file.close() 


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
     
    