# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
from dataflow import dataflow

import argparse


# NOTE: This is the main file, you propably don't need to change anything here

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Big Data image recomender",
    description="A project for recommending images based on similarities",
    epilog="Students project",
)

#parser.add_argument("mode", choices=["generator", "recomender"])
parser.add_argument("-f", "--folder", help="Path to folder containing the images")
parser.add_argument("-b", "--batch_size", action="store", default=500, help="Batch size for processing images")

# Parse the arguments from the command line
args = parser.parse_args()

# Switch control flow based on arguments
#if args.mode == "generator":

dataflow(args)
