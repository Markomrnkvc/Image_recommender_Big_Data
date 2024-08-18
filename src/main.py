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
import pandas as pd
from resnet_extraction import ResNet_Feature_Extractor

#picklefiley = "/Users/mjy/Downloads/data_clustered_5kentries.pkl"
#data = pd.read_pickle(picklefiley)
#print(data.head())

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Big Data image recomender",
    description="Set mode to either 'generator' or 'recommender'. If 'recommender', specify the method.",
    epilog="Students project",
)

parser.add_argument("-m", "--mode",choices=["generate", "cluster", "recommender"],  help="Choose which mode you want to execute")
parser.add_argument("-f", "--folder", help="Path to folder containing the images")
parser.add_argument("-me", "--method", nargs='+', choices=["histogram", "embeddings", "hashes"], help="choose which method you want to use for comparing the upload with the data base")

args = parser.parse_args()

# switch control flow based on arguments
if args.mode == "generate":
    print("generating features for the dataset...")
    dataflow(args)
    
elif args.mode == "cluster":
    print("clustering the dataset...")
    fit_cluster(n_clusters=30)
    
elif args.mode == "recommender" and args.method != None:
    print("starting recommendation app...")
    recommender = Recommender(methods=args.method)
    recommender.recommend()