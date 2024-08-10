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