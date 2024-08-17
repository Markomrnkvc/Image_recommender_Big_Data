# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
from dataflow import dataflow
from clustering import fit_cluster, predict_cluster

import argparse


# NOTE: This is the main file, you propably don't need to change anything here

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Big Data image recomender",
    description="A project for recommending images based on similarities",
    epilog="Students project",
)

#parser.add_argument("mode", choices=["generator", "recomender"])
#parser.add_argument("mode", choices=["generate", "cluster", "fit"])
parser.add_argument("-m", "--mode",choices=["generate", "cluster", "recommender"],  help="Choose which mode you want to execute")
parser.add_argument("-f", "--folder", help="Path to folder containing the images")
parser.add_argument("-me", "--method", choices=["histogram", "embeddings", "hashes"], help="choose which method you want to use to compare the new image with the data base")

parser.add_argument("-b", "--batch_size", action="store", default=500, help="Batch size for processing images")

#parser.add_argument("method", choices=["histogram", "embeddings", "hashes"])



# Parse the arguments from the command line
args = parser.parse_args()

# Switch control flow based on arguments

if args.mode == "generate":

    dataflow(args)
    
elif args.mode == "cluster":
    fit_cluster(n_clusters=10)
    
elif args.mode == "recommender" and args.method != None:
    import random
    from histograms import hist
    from phashes import perceptual_hashes
    import cv2
    
    image_path = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg"

    img = cv2.imread(image_path)
    #modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"
    #img = cv2.imread(image_path) 

    histogram = hist(img)

    embedding = random.randint(0,1000)

    phash_vector = perceptual_hashes(img)
    
    if args.method == "histogram":
        print(predict_cluster(img, image_path, args, data = histogram))
    elif args.method == "embeddings":
        print(predict_cluster(img, image_path, args, data = embedding))
    elif args.method == "hashes":
        print(predict_cluster(img, image_path, args, data = phash_vector))
    

#if args.mode == "cluster":#
 #   kmeans.py