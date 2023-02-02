import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

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