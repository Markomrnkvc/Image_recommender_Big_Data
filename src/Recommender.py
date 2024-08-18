import cv2
import os
import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog
from clustering import predict_cluster
from resnet_extraction import ResNet_Feature_Extractor
from phashes import perceptual_hashes
from histograms import hist
from scipy.spatial.distance import euclidean, hamming


class Recommender:
    global root
    root = os.getcwd()

    def __init__ (self, methods):
        self.methods = methods
        
    def filedialog(self):
        tk.Tk() #root = tk.Tk()
        #root.iconify() # no withdraw()
        source_image_paths = filedialog.askopenfilenames(title="upload your image(s)")
        if not source_image_paths:
            return None
        
        # turn image path(s) into arrays
        source_images = []
        for path in source_image_paths:
            img = cv2.imread(path)
            if img is not None:
                source_images.append(img)
            else:
                print(f"Failed to load image: {path}")
        
        return source_images
        
    def recommend(self):
        # open filedialog to select image(s):
        source_images = self.filedialog()

        if not source_images:
            print("No image selected.")
            return
        
        combined_results = []

        for method in self.methods:
            print(f"processing with method: {method}")

            # extract features from the uploaded image(s):
            features = [self.extract_features(img, method) for img in source_images]
            uploaded_feature = np.mean(features, axis=0)
            
            if uploaded_feature is not None:
                #cluster the upload image to get class number
                print("before clustering")
                #print(uploaded_feature)
                print(method)
                cluster = predict_cluster('unused_path', method, uploaded_feature)
                print(cluster)

                # load the features from pickle: filtered by matching cluster numbers
                pickle_path = os.path.join(root, "pickle/data_cluster.pkl")
                dataset = pd.read_pickle(pickle_path)
                
                #only get entries with the same cluster as the upload:
                if method == "embeddings" and method == "hashes" and method == "histogram":
                    print("in first if*++++++++")
                    clustered_dataset = dataset[dataset['cluster_embedding'] == cluster]
                elif method == "embeddings":
                    clustered_dataset = dataset[dataset['cluster_embedding'] == cluster]
                elif method == "hashes":
                    clustered_dataset = dataset[dataset['cluster_perceptual_hashes'] == cluster]
                elif method == "histogram":
                    clustered_dataset = dataset[dataset['cluster_histogram'] == cluster]

                nearest_neighbors = self.find_nearest_neighbors(uploaded_feature, clustered_dataset, method, k=5)
                combined_results.append((method, nearest_neighbors)) #combine top-k-images from each method
            else:
                print("Failed to extract features from the upload.")

        # display the combined results
        self.show_results(combined_results)
    
    def extract_features(self, img, method):
        if method == "embeddings":
            resnet_extractor = ResNet_Feature_Extractor()
            return resnet_extractor.extract_features(img)
        elif method == "hashes":
            return perceptual_hashes(img)
        elif method == "histogram":
            return hist(img)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def find_nearest_neighbors(self, uploaded_feature, dataset, method, k=5):
        distances = []

        # set method_column to fitting column-name according to the method used
        if method == "embeddings":
            method_column = 'Embeddings'
        elif method == "hashes":
            method_column = 'Perceptual_Hash'
        elif method == "histogram":
            method_column = 'RGB_Histogram'

        uploaded_feature = np.ravel(uploaded_feature) #convert uploaded_feature to a 1D array
        
        for image_id, feature in zip(dataset['Image_ID'],dataset[method_column]):
            feature = np.ravel(np.array(feature))
            if method == "embeddings":
                dist = euclidean(uploaded_feature, feature)
            elif method == "hashes":
                dist = hamming(uploaded_feature, feature) * len(feature)
            elif method == "histogram":
                dist = self.chi_square_distance(uploaded_feature, feature)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            distances.append((dist, image_id)) #store distance & image id

        #sort distances by the computed distance
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        top_images = []

        csv_path = os.path.join(root, "csv/images.csv")
        img_path_column = pd.read_csv(csv_path)

        for _, idx in top_k:
            print(idx)
            image_path = img_path_column[img_path_column['ID'] == image_id]['Name'].iloc[0] #get path of recommended image #df[df['Age'] == value]
            print(f"image_path: {image_path}")
            img = cv2.imread(image_path)
            if img is not None:
                top_images.append(img)

        return top_k, top_images

    
    def chi_square_distance(self, histA, histB, eps=1e-10):
        return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

    def show_results(self, combined_results):
        if not combined_results:
            print("No neighbors found.")
            return

        # calculate total number of images to display
        total_images = sum(len(top_images) for _, (_, top_images) in combined_results)
        
        # grid size for subplots (rows, columns)
        cols = 5 #bc 5 recommended images are demanded
        rows = (total_images + cols - 1) // cols
        plt.figure(figsize=(15, 5 * rows))  #figure size based on the number of rows
        plt.suptitle("We thought you might also like the following:", fontsize=16)

        img_idx = 1  #tracks the subplot index

        for method, (top_k, top_images) in combined_results:
            for i, (dist, img) in enumerate(zip(top_k, top_images)):
                plt.subplot(rows, cols, img_idx)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert BGR to RGB for correct color display
                plt.title(f"{method}: Dist: {dist[0]:.2f}")  # display method, distance in the title
                plt.axis('off')
                img_idx += 1

        plt.show()


#recommender = Recommender(method="resnet_embedding")
#recommender.recommend()