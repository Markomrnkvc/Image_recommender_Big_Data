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
import numba

start = time.time()

def calculate_mean(pixels_list):
    mean = 0
    total_pixels = len(pixels_list)
    for i in range(total_pixels):
        mean += pixels_list[i] / total_pixels
    print(f"mean: {mean}")
    return mean

def average_color(img):
    #plt.imshow(img)
    npmean = np.mean(img)
    
    print(f"np mean: {npmean}")
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


def make_bits_list(mean, pixels_list):
    bits_list = []
    for i in range(len(pixels_list)):
        if pixels_list[i] >= mean:
            bits_list.append(255)
        else:
            bits_list.append(0)
    return bits_list

def generate_hash(frame, hash_size = 16):
    frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
    frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
    pixels_list = grab_pixels(frame_squeezed)
    #mean_color = calculate_mean(pixels_list)
    npmean_color = average_color(frame)
    bits_list = make_bits_list(npmean_color, pixels_list)
    hashed_frame = hashify(frame_squeezed, bits_list)
    hashed_frame = cv2.cvtColor(hashed_frame, cv2.COLOR_GRAY2BGR)
    return bits_list, hashed_frame


img = cv2.imread("D:/data/image_data/Landscapes/00000362_(6).jpg")#"D:\data\image_data\Landscapes\00000021_(6).jpg"
img_tocompare = cv2.imread("D:/data/image_data/Landscapes/00000363_(6).jpg")
#img = cv2.imread("C:/Users/marko/Documents/viertes_semester/BigData/Image_recommender_Big_Data/src/images/000000000024.jpg")

bits_list, hashed_frame = generate_hash(img)
bits_list2, hashed_frame2 = generate_hash(img_tocompare)

print(f" hamming: {hamming(bits_list, bits_list2)}")
#print(bits_list)

print(bits_list)


#measuring time
end = time.time()
print(f" runtime: {end - start}")

fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 columns

# Display the first image in the first subplot
ax1.imshow(hashed_frame)
ax1.axis('off')  # Optional: turns off the axis

# Display the second image in the second subplot
ax2.imshow(hashed_frame2)
ax2.axis('off')  # Optional: turns off the axis

# Show the plot
plt.show()

plt.imshow(hashed_frame)
plt.imshow(hashed_frame2)