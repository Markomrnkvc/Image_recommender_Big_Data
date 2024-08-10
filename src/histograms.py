# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 02:32:15 2024

@author: marko
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

img = cv.imread("D:\data\image_data\extra_collection\electronics\jeswin-thomas-dfRrpfYD8Iw-unsplash.jpg")

def hist(img):
    """Function which calculates the histogram for RGB images
    and returns a flattened histogram vector including all three channels"""
    
    color = ('b','g','r')
    for i,col in enumerate(color):
        histogram = cv.calcHist([img],[i],None,[256],[0,256])
        
        
        histogram = histogram.flatten()
        
    return histogram, col


"""

das kann eig alles weg:
    
def plot_hist(histr, col):
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()

histr, col = hist(img)
histr1, col1 = hist(img)

histr = histr.flatten()
histr1 = histr1.flatten()

print(cosine_similarity([histr], [histr1]))




muss nur noch hinzufügen, dass der vektor dann in die files geladen wird

später brauchen wir noch vergleichfunktonen wie cosine (kann man von sklearn übernehmen [siehe imports] )
"""