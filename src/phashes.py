# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:05:23 2024

@author: marko
"""
import cv2
from scipy.spatial.distance import hamming
from matplotlib import pyplot as plt
import PIL
import time
import numpy as np

start = time.time()
def perceptual_hashes(img, return_img = False):
    def calculate_mean(pixels_list):
        mean = 0
        total_pixels = len(pixels_list)
        for i in range(total_pixels):
            mean += pixels_list[i] / total_pixels
        return mean
    
    def average_color(img):
        #plt.imshow(img)
        npmean = np.mean(img)
        
        return npmean
    
    def grab_pixels(squeezed_frame):
        pixels_list = []
        for x in range(0, squeezed_frame.shape[1], 1):
            for y in range(0, squeezed_frame.shape[0], 1):
                pixel_color = squeezed_frame[x, y]
                pixels_list.append(pixel_color)
        return pixels_list
    
    
    def hashify(squeezed_frame, bits_list):
        bit_index = 0
        hashed_frame = squeezed_frame
        for x in range(0, squeezed_frame.shape[1], 1):
            for y in range(0, squeezed_frame.shape[0], 1):
                hashed_frame[x, y] = bits_list[bit_index]
                bit_index += 1
        return hashed_frame
    
    
    def make_bits_list(mean, pixels_list, max_pixelvalue):
        bits_list = []
        for i in range(len(pixels_list)):
            if pixels_list[i] >= mean:
                bits_list.append(max_pixelvalue)
            else:
                bits_list.append(0)
        return bits_list
    if return_img == True:
        """
        combining all methods into one
        including img version after hashing for plotting
        
        """
        def generate_hash(frame, hash_size = 16):
            frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
            frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
            pixels_list = grab_pixels(frame_squeezed)
            #mean_color = calculate_mean(pixels_list)
            npmean_color = average_color(frame)
            bits_list = make_bits_list(npmean_color, pixels_list, max_pixelvalue=255)
            hashed_frame = hashify(frame_squeezed, bits_list)
            hashed_frame = cv2.cvtColor(hashed_frame, cv2.COLOR_GRAY2BGR)
            return bits_list, hashed_frame
    
        phash_vector, hashed_img = generate_hash(img)
    
        return phash_vector, hashed_img

    elif return_img == False:
    
        """
        combining all methods into one
        
        """
        def generate_hash(frame, hash_size = 16):
            frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
            frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
            pixels_list = grab_pixels(frame_squeezed)
            #mean_color = calculate_mean(pixels_list)
            npmean_color = average_color(frame)
            bits_list = make_bits_list(npmean_color, pixels_list, max_pixelvalue = 1)
            
            
            return bits_list
    
        phash_vector = generate_hash(img)
        
        return phash_vector
    

"""
Zum ausprobieren:
"""
img = cv2.imread("D:/data/image_data/Landscapes/00000362_(6).jpg")#"D:\data\image_data\Landscapes\00000021_(6).jpg"
img_tocompare = cv2.imread("D:/data/image_data/Landscapes/00000363_(6).jpg")
#img = cv2.imread("C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg")
"""
phash_vector, hashed_img = perceptual_hashes(img, return_img=True)
phash_vector2, hashed_img2 = perceptual_hashes(img_tocompare, return_img=True)
"""
phash_vector = perceptual_hashes(img)
phash_vector2= perceptual_hashes(img_tocompare)

print(f" hamming: {hamming(phash_vector, phash_vector2)}")
#print(bits_list)

#print(phash_vector)


#measuring time
end = time.time()
print(f" runtime: {end - start}")
"""
fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 columns

# Display the first image in the first subplot
ax1.imshow(hashed_img)
ax1.axis('off')  # Optional: turns off the axis

# Display the second image in the second subplot
ax2.imshow(hashed_img2)
ax2.axis('off')  # Optional: turns off the axis

# Show the plot
plt.show()

plt.imshow(hashed_img)
plt.imshow(hashed_img2)
"""