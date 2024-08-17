# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
from dataflow import dataflow
from Recommender import Recommender
from clustering import fit_cluster, predict_cluster
import argparse
import os
from Resnet_Extraction import ResNet_Feature_Extractor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Big Data image recomender",
    description="Set mode to either 'generator' or 'recommender'. If 'recommender', specify the method.",
    epilog="Students project",
)

#parser.add_argument("mode", choices=["generator", "recomender"])
#parser.add_argument("mode", choices=["generate", "cluster", "fit"])
parser.add_argument("-m", "--mode",choices=["generate", "cluster", "recommender"],  help="Choose which mode you want to execute")
parser.add_argument("-f", "--folder", help="Path to folder containing the images")
parser.add_argument("-me", "--method", choices=["histogram", "embeddings", "hashes"], help="choose which method you want to use to compare the new image with the data base")

parser.add_argument("-b", "--batch_size", action="store", default=500, help="Batch size for processing images")

#parser.add_argument("method", choices=["histogram", "embeddings", "hashes"])



#if 'mode' is 'recommender':
parser.add_argument('--method', nargs='+', choices=['resnet_embedding', 'phash_vector', 'histogram'], required=True, help='Specify one or more methods')

args = parser.parse_args()

# Switch control flow based on arguments

if args.mode == "generate":
    print("generating features for the dataset...")
    dataflow(args)
    
elif args.mode == "cluster":
    fit_cluster(n_clusters=10)
    
elif args.mode == "recommender" and args.method != None:

    recommender = Recommender(methods=args.method)
    recommender.recommend()
    import random
    from histograms import hist
    from phashes import perceptual_hashes
    import cv2
    
    image_path = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg"

    img = cv2.imread(image_path)
    #modelfile = "C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/pickle/kmeans_model.pkl"
    #img = cv2.imread(image_path) 

    histogram = hist(img)

    extractor = ResNet_Feature_Extractor(model_weights="imagenet")
    resnet_embedding = extractor.extract_features(img)

    phash_vector = perceptual_hashes(img)
    
    if args.method == "histogram":
        print(predict_cluster(img, image_path, args, data = histogram))
    elif args.method == "embeddings":
        print(predict_cluster(img, image_path, args, data = embedding))
    elif args.method == "hashes":
        print(predict_cluster(img, image_path, args, data = phash_vector))
    
