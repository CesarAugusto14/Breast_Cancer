""" This code is made by Cesar Augusto Sanchez, to resize the images from the folder
train_images_jpeg to be 1024 on their longest side, and save them in the folder named 1024_images.

This code is not doing further preprocessing on any of the images, so the images will be raw and without
human intervention. Therefore, the aspect of the breasts should be preseved.


"""
import cv2
import os
from tqdm import tqdm

# Path to the folder with the original images in JPEG format
path = './data/train_images_jpeg/'
if not os.path.exists('./data/1024_images/'):
    os.mkdir('./data/1024_images/')
else:
    print('The folder 1024_images alread exists.')

# For each patient, we will need to create a new image folder within the 1024_images folder.
for patient in tqdm(sorted(os.listdir(path))):
    # Loop for each patient. We need fo ignore the .DS_Store file if macOS.
    if patient != '.DS_Store':
        # Create the folder for the jpeg images.
        if not os.path.exists('./data/1024_images/' + str(patient)):
            os.mkdir('./data/1024_images/' + str(patient))
        else:
            print('The folder 1024_images/' +
                  str(patient) + ' already exists.')

        path_to_save = './data/1024_images/' + patient + '/'
        for image in sorted(os.listdir(path + str(patient))):
            # Loop for each image. We need fo ignore the .DS_Store file if macOS.
            if image != '.DS_Store':
                # Saving the image in JPEG format trying to preserve the original information.
                data = cv2.imread(path + patient + '/' + image)
                # We need to resize the image to be 1024 on its longest side.
                if data.shape[0] > data.shape[1]:
                    # This is the case when the image is taller than wide.
                    new_shape = (int(data.shape[1] * 1024/data.shape[0]), 1024)
                else:
                    # This is the case when the image is wider than tall.
                    new_shape = (1024, int(data.shape[0] * 1024/data.shape[1]))
                data = cv2.resize(data, new_shape, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(path_to_save + image, data)
