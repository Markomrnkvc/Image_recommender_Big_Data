# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:12:17 2024

@author: marko
"""
import os
import pandas as pd
from random import randint


def create_pk(pk_path):

    if os.path.exists(pk_path) == False:
        # naming columns
        cols = [
            "Image_ID",
            "Embeddings",
            "RGB_Histogram",
            "Perceptual_Hash",
        ]  # 'Height', 'Width', 'Channels', 'Avg Blue', 'Avg Red', 'Avg Green',

        # creating DataFrame
        df = pd.DataFrame(columns=cols)
        # saving to pickle
        pd.to_pickle(df, pk_path)


def save_in_df(embedding, image_id, histogram, phash_vector, df):  # h, w, c, avg_color,

    # opening pickle file
    # df = pd.read_pickle(path)
    # adding new data to DataFrame
    df.loc[len(df)] = [
        int(image_id),
        embedding,
        histogram,
        phash_vector,
    ]  # int(h), int(w), c, avg_color[0], avg_color[1],  avg_color[2],
