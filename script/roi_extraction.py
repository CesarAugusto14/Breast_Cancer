"""
author: @cesarasa

This code is made by cesar sanchez-villalobos to:

1st: find the ROI using several methods.
2nd: crop the ROI.
3rd: resize the ROI to 256x256.

In the code, I will define several methods to find the ROI, according to: 

https://www.kaggle.com/code/awsaf49/rsna-bcd-roi-methods-comparison

The methods are:

- Non-zero finding of the ROI.
- Binary Erode Largest Contour (BELC) method.
- Average Pixel Value (APV) method.
- Binary Masked Largest Contour (BMLC) method.
- Standard Deviation (STD) method.
- Blur Binary Largest Contour (BBMLC) method.
"""

import cv2
import os
import numpy as np

def find_ROI_non_zero(image_path):
    """
    This function uses the non-zero method to find the ROI.

    The function will be used for grayscale images from the Breast-Cancer Dataset. 

    Args:
        image_path: the path to the image.

    Returns:
        The cropped and resized image. 
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_OTSU)[1]
    # Find the non-zero pixels
    ys, xs = np.nonzero(bin_img)
    # Find the bounding box
    roi = img[min(ys):max(ys), min(xs):max(xs)]
    # Resize
    roi_image = cv2.resize(roi, (256, 256))
    return roi, roi_image, ys, xs

def find_ROI_BELC(image_path):
    """
    This function uses the BELC method to find the ROI.

    The function will be used for grayscale images from the Breast-Cancer Dataset. 

    Args:
        image_path: the path to the image.

    Returns:
        The cropped and resized image. 
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    bin_img = cv2.erode(bin_img, np.ones((11,11)))
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    # Find the non-zero pixels
    xs = contour.squeeze()[:, 0]
    ys = contour.squeeze()[:, 1]
    # Find the bounding box
    roi = img[min(ys):max(ys), min(xs):max(xs)]
    # Resize
    roi_image = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_CUBIC)
    return roi, roi_image, ys, xs

def find_ROI_BBLC(image_path, size = (256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian Blurring:
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Thresholding:
    ret, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Finding Contours:
    contours, _ = cv2.findContours(breast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find max contour:
    max_contour = max(contours, key=cv2.contourArea)
    # Find bounding box:
    x, y, w, h = cv2.boundingRect(max_contour)
    # Crop image:
    roi = img[y:y+h, x:x+w]
    # Resize
    roi_image = cv2.resize(roi, size, interpolation=cv2.INTER_CUBIC)
    return roi, roi_image, y, x, w, h