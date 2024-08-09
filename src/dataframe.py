
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
        #naming columns
        cols = ['Image_ID', 'Embeddings', 'Height', 'Width', 'Channels', 'Avg Blue', 'Avg Red', 'Avg Green']
        #creating DataFrame
        df = pd.DataFrame(columns=cols)
        #saving to pickle
        df.to_pickle(pk_path)
        
def save_in_df(embedding, image_id, h, w, c, avg_color, df):
    #opening pickle file
    #df = pd.read_pickle(path)
    
    embedding=randint(0,1000) #need placeholder value, no embeddings yet
    
    #adding new data to DataFrame
    df.loc[len(df)] = [int(image_id), embedding, int(h), int(w), c, 
		avg_color[0], avg_color[1], 
		avg_color[2]]
    
    
    
    
    