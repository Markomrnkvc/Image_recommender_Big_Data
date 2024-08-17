# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
from dataflow import dataflow
from Recommender import Recommender
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Big Data image recomender",
    description="Set mode to either 'generator' or 'recommender'. If 'recommender', specify the method.",
    epilog="Students project",
)

parser.add_argument("-f", "--folder", help="Path to SSD-folder containing the images")
parser.add_argument("mode", choices=["generator", "recommender"], help="Choose the mode: 'generator' to generate features, or 'recommender' for recommendations.")

#if 'mode' is 'recommender':
parser.add_argument('--method', nargs='+', choices=['resnet_embedding', 'phash_vector', 'histogram'], required=True, help='Specify one or more methods')

args = parser.parse_args()

if args.mode == "recommender":
        if args.method is None:
            parser.error("The 'recommender' mode requires the '--method' argument to be specified.")
        recommender = Recommender(methods=args.method)
        recommender.recommend()
elif args.mode == "generator":
        print("generating features for the dataset...")
        dataflow(args)


"""TO DO:
        data is not getting loaded into pickle file 
        IDs of data in csv are wrong """