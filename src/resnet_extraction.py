import cv2
import numpy as np

from tqdm import tqdm
from generator import image_generator

from keras import Sequential
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.layers import GlobalMaxPool2D
from keras.utils import img_to_array


class ResNet_Feature_Extractor:
    """
    This class is used to build the ResNet model, preprocess the images from the data set
    and then extracting features out of the image-files based on the pre-trained resnet model.
    The features and the image-paths get saved in a separate pickle file, with matching IDs.
    """

    def __init__(self, model_weights="imagenet", input_shape=(224, 224, 3)):
        self.model = self.build_model(model_weights, input_shape)

    def build_model(self, weights, input_shape):
        """instantiates and returns the ResNet50 model."""

        base_model = ResNet50(
            weights=weights, include_top=False, input_shape=input_shape
        )
        base_model.trainable = False
        model = Sequential([base_model, GlobalMaxPool2D()])
        model.summary()
        return model

    def image_preprocessing(self, img):
        """preprocesses the image, extracts its features, returns the feature vector."""

        # print(f"Processing file: {image_generator.image_path}")
        try:
            if (
                img.shape[2] == 1
            ):  # falls das Bild nur 1 Kanal hat, konvertiere es in ein 3-Kanal-Bild
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (224, 224))  # resize
            img_arr = img_to_array(img)  # convert to array
            expand_img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
            pre_pr_img = preprocess_input(expand_img_arr)  # Preprocess the image
            result = self.model.predict(
                pre_pr_img
            ).flatten()  # Predict and flatten the output
            normal_result = result / np.linalg.norm(result)  # Normalize the result
            return normal_result

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def extract_features(self, img):
        """extracts all features from the image and puts them into the "feature_list"
        which is then returned into the pickle file in the dataflow that includes all features from all embeddings.
        """

        print(f"extracting features...")
        feature = self.image_preprocessing(img)
        print(
            f"this is the FEATURE this is the FEATURE this is the FEATURE this is the FEATURE: {feature}"
        )
        return feature


# ------------------------- MAIN -------------------------

"""def main():
    extraction = ResNet_Feature_Extractor()

    # extract features and save them to pickle files:
    extraction.extract_features()



if __name__ == "__main__":
    main()"""
