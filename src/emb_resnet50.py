import numpy as np
import os
import pickle
import shutil
from tqdm import tqdm

from scipy.stats import norm
from scipy.spatial.distance import euclidean

import tkinter as tk
from tkinter import filedialog

from keras import Sequential
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.layers import GlobalMaxPool2D
from keras.utils import load_img, img_to_array


# ---------------------------------------------PROCESSING IMAGES, EXTRACTING FEATURES------------------------------------------------
# defining the model:
model = ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable = False
model = Sequential([model, GlobalMaxPool2D()])
model.summary()

# SSD-path:
path = r'/Volumes/ExtremeSSD/data/image_data'

# allowed file extensions:
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# check if it's the right file type
def is_image_file(filename):
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)

# get all image-paths:
image_files = []
def find_all_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    return image_files
all_images = find_all_images(path)
images = all_images[:700]   # limit to the first 500 images for testing

# resize images, turn into arrays, 
def image_preprocessing(path,model):
    print(f"Processing file: {path}")

    try: 
        img = load_img(path, target_size=(224,224)) # load & resize
        img_arr = img_to_array(img) # convert to array
        expand_img_arr = np.expand_dims(img_arr, axis=0) # --> output: (1, 224, 224, 3), is needed but idk why
        pre_pr_img = preprocess_input(expand_img_arr) # preprocess
        result = model.predict(pre_pr_img).flatten() # predict and flatten the output
        normal_result = result/np.linalg.norm(result) # normalize model
        return normal_result
    
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return None


pickle.dump(images, open('pickle/emb_images.pkl','wb'))
feature_list=[]
for file in tqdm(images):
    feature_list.append(image_preprocessing(file, model))
# saves feature vector to according image path
pickle.dump(feature_list,open('pickle/features.pkl','wb'))

# load the stored features and images:
print(f"current directory is: {os.getcwd()}")
image_paths = pickle.load(open(r'pickle/emb_images.pkl','rb'))
feature_list = (pickle.load(open(r'pickle/features.pkl','rb')))

# create filtered lists that don't keep the None values:
filtered_feature_list = []
filtered_image_paths = []

for feature, path in zip(feature_list, image_paths):
    if feature is not None:
        filtered_feature_list.append(feature)
        filtered_image_paths.append(path)


# -------------------- HANDLING THE UPLOAD IMAGE --------------------

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

# create a root window (will NOT be shown)
root = tk.Tk()
root.withdraw()  # hide the window

# open a filedialog (to select file from Finder etc.):
source_image_path = filedialog.askopenfilename(title="Upload an image")


# -------------------- SEARCH & SHOW RECOMMENDED IMAGES --------------------

# extract features from uploaded image as before in the preprocessing
def extract_features(image_path, model):
    return image_preprocessing(image_path, model)

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