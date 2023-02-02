"""
This script was created by Cesar Augusto Sanchez to convert all the DCM images from the folder
data/train_images/ to the folder data/train_images_jpeg/ in the format JPEG.
"""
import cv2
import os
import numpy as np
import pydicom
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut


# Path to the folder with the original images in DCM format
path = './data/train_images/'
os.mkdir('./data/train_images_jpeg/')

# Function for reading the image:
def load_image_pydicom(img_path, voi_lut=True):
    # Code made with help from https://www.kaggle.com/code/raddar/convert-dicom-to-np-array-the-correct-way
    # and https://www.kaggle.com/code/bobdegraaf/dicomsdl-voi-lut
    dicom = pydicom.dcmread(img_path)
    img = dicom.pixel_array
    if voi_lut:
        img = apply_voi_lut(img, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        # The image is inverted when MONOCHROME1 is True. 
        img = np.max(img) - img
    
    # Normalize the image. First we need to subtract the minimum, then divide by the maximum and,
    # finally, multiply by 255 to get the image in the range 0-255. Set it to be uint8 to save memory.
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    return img

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
                data = load_image_pydicom(path + patient + '/' + image, voi_lut=True)
                cv2.imwrite(path_to_save + image[:-4] + '.jpg', data)
