"""
This script was created by Cesar Augusto Sanchez to convert all the DCM images from the folder
data/train_images/ to the folder data/train_images_jpeg/ in the format JPEG.
"""

import os
import numpy as np
import pydicom
from tqdm import tqdm
import cv2

# Path to the folder with the original images in DCM format
path = './data/train_images/'
os.mkdir('./data/train_images_jpeg/')


for patient in tqdm(sorted(os.listdir(path))):
    # Loop for each patient. We need fo ignore the .DS_Store file if macOS.
    if patient != '.DS_Store':

        # Create the folder for the jpeg images.
        os.mkdir('./data/train_images_jpeg/' + str(patient))
        path_to_save = './data/train_images_jpeg/' + patient + '/'

        for image in sorted(os.listdir(path + str(patient))):
            # Loop for each image. We need fo ignore the .DS_Store file if macOS.

            if image != '.DS_Store':
                # Saving the image in JPEG format trying to preserve the original information.
                data = pydicom.dcmread(path + patient + '/' + image)
                data = data.pixel_array
                data = data - np.min(data)
                data = data/np.max(data)
                data = (data * 255).astype(np.uint8)
                cv2.imwrite(path_to_save + image[:-4] + '.jpg', data)
