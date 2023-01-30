""" This code is made by Cesar Augusto Sanchez, to resize the images from the folder
train_images_jpeg to be 512x512, and save them in the folder named 512_images.

This code is not doing further preprocessing on any of the images, so the images will be raw and without
human intervention. Therefore, the aspect of the breasts should be preseved.


"""
import cv2
import os
from tqdm import tqdm

# Path to the folder with the original images in JPEG format
path = './data/train_images_jpeg/'
if not os.path.exists('./data/512_images/'):
    os.mkdir('./data/512_images/')
else:
    print('The folder 512_images already exists.')

# For each patient, we will need to create a new image folder within the 512_images folder.
for patient in tqdm(sorted(os.listdir(path))):
    # Loop for each patient. We need fo ignore the .DS_Store file if macOS.
     if patient != '.DS_Store':
        # Create the folder for the jpeg images.
        if not os.path.exists('./data/512_images/' + str(patient)):
            os.mkdir('./data/512_images/' + str(patient))
        else:
            print('The folder 512_images/' + str(patient) +
                   ' already exists.')
        
        path_to_save = './data/512_images/' + patient + '/'
        for image in sorted(os.listdir(path + str(patient))):
            # Loop for each image. We need fo ignore the .DS_Store file if macOS.
            if image != '.DS_Store':
                data = cv2.imread(path + patient + '/' + image)
                data = cv2.resize(data, [512,512], interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(path_to_save + image, data)