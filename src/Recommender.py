import os
import shutil
import tkinter as tk
from tkinter import filedialog


class Recommender:

    def __init__ (self, upload_img, method):
        self.upload_img = upload_img
        self.method = method

    def save_upload_img(source_path, upload_dir='uploads'):
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
    
        try:
            filename = os.path.basename(source_path)
            upload_path = os.path.join(upload_dir, filename)
            shutil.copy(source_path, upload_path)   # copy file from one place to another
            return upload_path ##man könnte evtl statt den path direkt das bild/die bilder als arr übergeben.
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def recommend(self, args.method):

        """sets the method for recommending based on the chosen method."""

        if args.method == "resnet":
            return Comparer_ResNet()
        elif args.method == "phashes":
            return Comparer_Phashes()
        elif args.method == "histograms":
            return Comparer_Histograms()
        else:
            raise ValueError(f"unknown method: {args.method}")
        
class Comparer_ResNet():
    pass

class Comparer_Phashes():
    pass

class Comparer_Histograms():
    pass

    def save_upload_img(source_path, upload_dir='uploads'):
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
    
        try:
            filename = os.path.basename(source_path)
            upload_path = os.path.join(upload_dir, filename)
            shutil.copy(source_path, upload_path)   # copy file from one place to another
            return upload_path
        except Exception as e:
            print(f"Error: {e}")
            return None

    def filedialog():

        # create a root window (will NOT be shown)
        root = tk.Tk()
        root.withdraw()  # hide the window

        # open a filedialog (to select file from Finder etc.):
        source_image_path = filedialog.askopenfilename(title="Upload an image")


        # -------------------- SEARCH & SHOW RECOMMENDED IMAGES --------------------

    # extract features from uploaded image as before in the preprocessing
    def extract_features(upload_img):
        return image_preprocessing(upload_img)

    # find 10 NN by measuring the euclidean distance between the feature-vectors
    def find_nearest_neighbors(uploaded_feature, filtered_feature_list, filtered_image_paths, k=10):
        distances = []
        for idx, feature in enumerate(filtered_feature_list):
            dist = euclidean(uploaded_feature, feature)
            distances.append((dist, filtered_image_paths[idx]))
    
        # sort by distance
        distances.sort(key=lambda x: x[0])
    
        # return the k closest neighbors
        return distances[:k]

    if source_image_path:
        saved_image_path = save_upload_img(source_image_path)   # save uploaded image

        if saved_image_path:
            print(f"Image successfully saved at: {saved_image_path}")

            # extract features from uploaded image:
            uploaded_feature = image_preprocessing(saved_image_path, model)
            if uploaded_feature is not None:
                # find NN
                print(f"Uploaded feature shape: {uploaded_feature.shape}")
                if feature_list:
                    print(f"Sample feature : {feature_list}")
                nearest_neighbors = find_nearest_neighbors(uploaded_feature, filtered_feature_list, filtered_image_paths, k=10)
                print("10 Nearest Neighbors:")
                for dist, img_path in nearest_neighbors:
                    print(f"Image: {img_path}, Distance: {dist}")
            else:
                print("Failed to extract features from the uploaded image.")
        else:
            print("Failed to save the image.")