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
import numpy
import cv2
import argparse
import numpy
import csv
from tqdm import tqdm
import numpy as np

# ID for each image, refered to in csv-file
image_id = 0

# path to csv-file
csv_file = "csv\images.csv"  # "C:/Users/marko/OneDrive/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/csv/images.csv"
error_file = "csv\error_images.csv"
current_path = os.getcwd()
csv_path = join(current_path, csv_file)
error_path = join(current_path, error_file)


def create_csv(args, csv_path):
    # Check whether the CSV
    # exists or not if not then create one.

    # creating csv if not existing
    if os.path.exists(csv_path) == False:
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(["ID", "Name"])

    # creating csv if not existing
    if os.path.exists(error_path) == False:
        with open(error_path, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(["Name"])


def image_generator(args):

    path = Path(args.folder)

    # creating a list with all paths already loaded into csv
    list_img = []
    error_list_img = []
    current_ID = -1
    # C:\Users\marko\Documents\viertes_semester\BigData\Image_recommender_Big_Data\src\csv
    # C:/Users/marko/Documents/viertes_Semester/Big_Data/Image_recommender_Big_Data/src/
    with open(csv_path, mode="r") as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            list_img.append(lines[1])
            current_ID += 1

    with open("csv/error_images.csv", mode="r") as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            error_list_img.append(lines[0])
            current_ID += 1

    gen_uptodate = False  # variable we use to check if the generator is yielding new images or old ones

    # generator that runs image files from our given directory as the parameter
    for root, _, files in os.walk(path):

        for file in tqdm(files, total=444880, initial=current_ID):
            # for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                image_path = os.path.join(root, file)

                if gen_uptodate == False:  # generator still yielding old images
                    # checking if image is already in database
                    if image_path not in list_img and image_path not in error_list_img:
                        # set to True if one image has not been added to csv yet
                        gen_uptodate = True

                        print(f"\ngen_uptodate set to {gen_uptodate}\n")

                if gen_uptodate == True:  # if one is new, all folowing will be new too
                    # print("new image loaded into csv")
                    # loading the image
                    img = cv2.imread(image_path)

                    # print(image_path)
                    # print(current_ID)
                    image_id = current_ID
                    # print(img)
                    yield img, image_path, image_id
                    # setting ID counter up
                    current_ID += 1
                    # print(current_ID)


def data_writer(
    image_id, image_path, csv_path
):  # , h, w, c, avg_color, histogram, phash_vector

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image_id, image_path])
    file.close()
