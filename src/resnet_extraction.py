import numpy as np
import os
import pickle
from tqdm import tqdm

from scipy.spatial.distance import euclidean

from keras import Sequential
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.layers import GlobalMaxPool2D
from keras.utils import load_img, img_to_array


class ResNet_Feature_Extractor:
    """
    This class is used to build the ResNet model, preprocess the images from the data set
    and then extracting features out of the image-files based on the pre-trained resnet model.
    The features and the image-paths get saved in a separate pickle file, with matching IDs."""

    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    def __init__(self, model_weights='imagenet', input_shape=(224, 224, 3), SSD_path=r'/Volumes/ExtremeSSD/data/image_data'):
        self.model = self.build_model(model_weights, input_shape)
        self.base_path = SSD_path
        self.image_paths = self.find_all_images(self.SSD_path)
        self.feature_list = []

    def build_model(self, weights, input_shape):
        """instantiates and returns the ResNet50 model."""
        model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
        model.trainable = False
        model = Sequential([model, GlobalMaxPool2D()])
        model.summary()
        return model
    
    def is_image_file(self, filename):
        """check if it's the right file type"""
        return filename.lower().endswith(self.SUPPORTED_EXTENSIONS)
    
    def find_all_images(self, directory):
        """finds all image files in the specified directory and its subdirectories."""
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if self.is_image_file(file):
                    file_path = os.path.join(root, file)
                    image_files.append(file_path)
        return image_files 
    
    def image_preprocessing(self, path):
        """preprocesses the image, extracts its features, returns the feature vector."""
        print(f"Processing file: {path}")
        try:
            img = load_img(path, target_size=(224, 224))  # Load & resize
            img_arr = img_to_array(img)  # Convert to array
            expand_img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
            pre_pr_img = preprocess_input(expand_img_arr)  # Preprocess the image
            result = self.model.predict(pre_pr_img).flatten()  # Predict and flatten the output
            normal_result = result / np.linalg.norm(result)  # Normalize the result
            return normal_result
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            return None
        
    def extract_features(self):
        """extracts features from all images and saves them."""
        print(f"Extracting features for {len(self.image_paths)} images.")
        for file in tqdm(self.image_paths):
            feature = self.image_preprocessing(file)
            if feature is not None:
                self.feature_list.append(feature)
        
        self.save_pickle('pickle/image_paths.pkl', self.image_paths)
        self.save_pickle('pickle/features.pkl', self.feature_list)

    def save_pickle(self, filename, data):
        """saves the given data to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"data saved to {filename}")



# ------------------------- MAIN -------------------------

def main():
    extraction = ResNet_Feature_Extractor()

    # extract features and save them to pickle files:
    extraction.extract_features()



if __name__ == "__main__":
    main()