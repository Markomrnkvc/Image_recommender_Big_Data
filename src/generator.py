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
#from Pil import Image
#from main import args

#path of csv file for saving metadata
#my_file = args.folder #Path("C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/image_recomender/csv/images.csv") 

#ID for each image, refered to in csv-file
image_id = 0
#path to csv-file
csv_path = "C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/image_recomender/csv/images.csv"


def create_csv(args, csv_path):
    # Check whether the CSV 
    # exists or not if not then create one. 
    my_images = args.folder #Path("C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/image_recomender/csv/images.csv") 

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
            
def image_generator(args, path = Path("C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/image_recomender/images")):
    #global.image_id = 0
    if args.folder == True:
        path = Path(args.folder)
    # generator that runs image files from our given directory as the parameter
    for root, _, files in os.walk(path):
        
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                global image_id 
                image_id += 1
                #print(image_path, image_id)
                yield image_path, image_id
                
def get_data(args, image_path, image_id, csv_path):
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
    img = cv2.imread(image_path) 
    #print(img)
    h,w,c = img.shape
    
    """img_o = Image.open(image_path)
    form = img_o.format"""
       
    avg_color_per_row = numpy.average(img, axis = 0) 
    avg_color = numpy.average(avg_color_per_row, axis = 0) 
    #img_name = os.path.basename(image_path) #only img name, without whole path
   	
    with open(csv_path, 'a', newline = '') as file: 
   		writer = csv.writer(file) 
   		writer.writerow([image_id, image_path, h, w, c, 
   						avg_color[0], avg_color[1], 
   						avg_color[2]]) 
    file.close() 
   
    
"""files = img_loading_generator()
print(files)"""

def generate(args):
    """create_csv(args, csv_path)
    print("create")
    print(image_generator(args))
    image_path, image_id = next(image_generator(args))
    print("generator")
    get_data(args, image_path, image_id, csv_path)
    print("get data")"""
    create_csv(args, csv_path)
    #print("create")
    #print(image_generator(args))
    print(next(image_generator(args)))
    
    #image_path, image_id = next(image_generator(args))
    #print(image_path)
    for image_path, image_id in image_generator(args):
        print(image_id)
        #print(f" 1    :{image_path}")
        #print(image_generator(args))
        #image_path, image_id = next(image_generator(args))
        #print("generator")
        get_data(args, image_path, image_id, csv_path)
        print("get data")   
    
    
    