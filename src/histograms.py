# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 02:32:15 2024

@author: marko
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


    
def hist(img):
    
    """Function which calculates the histogram for RGB images
    and returns a flattened histogram vector including all three channels"""
    def calchist(img):
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        
        hist_r = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_b = cv.calcHist([img], [2], None, [256], [0, 256]).flatten()
        
        return hist_r, hist_g ,hist_b
    
    hist_r, hist_g ,hist_b = calchist(img)

    #normalisieren der Daten
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()
    
    #vereinen der Vektoren
    hist = np.concatenate([hist_r, hist_g ,hist_b])
    
    return hist


