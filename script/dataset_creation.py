import os
import matplotlib.pyplot as plt
import numpy as np
import torch

class DatasetBreastWrong(torch.utils.data.Dataset):
    """The labels will be available through df.cancer, whereas the images will be loaded from the /data/256_images folder.
    """
    def __init__(self, df, path_patients, transform=None):
        self.df = df
        self.path_patients = path_patients
        self.transform = transform
        self.list_patients = sorted(os.listdir(path_patients))
        self.list_patients = [x for x in self.list_patients if x != '.DS_Store']
        # From the list of patients, I will need to get access to the images
        self.list_images = []
        for patient in self.list_patients:
            for image in sorted(os.listdir(path_images + patient)):
                if image != '.DS_Store':
                    self.list_images.append(patient + '/' + image)
        self.list_images = sorted(self.list_images)
        self.labels = self.df.cancer.values
        self.labels = self.labels.astype(np.float32)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        path_image = self.path_patients + self.list_images[idx]
        image = plt.imread(path_image)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        # Get the label it should be either 0 or 1
        label = self.labels[idx]
        image, labels = image.cuda(), labels.cuda()
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class DatasetBreastDownsample(torch.utils.data.Dataset):
    """The labels will be available through df.cancer, whereas the images will be loaded from the /data/256_images folder.
    """
    def __init__(self, df, path_patients, transform=None, view='CC', breast = 'L'):
        # From the dataframe, I will need to remove the images that are not the view I want. 
        # I will also need to remove the images that are not the breast I want.
        # Finally, I need to downsample the majority class. 
        self.df = df
        self.path_patients = path_patients
        self.transform = transform
        self.view = view
        self.breast = breast
         # Working with the dataframe
        self.df = self.df[self.df.view == self.view]
        self.df = self.df[self.df.laterality == self.breast]

        # I need to randomly select the patients that are in the majority class
        self.list_patients_majority = np.unique(self.df.patient_id[(self.df.cancer == 0).values])
        self.list_patients_minority = np.unique(self.df.patient_id[(self.df.cancer == 1).values])
        # I will need to downsample the majority class

        self.list_patients_majority = np.random.choice(self.list_patients_majority, 
                                                       size=len(self.list_patients_minority), 
                                                       replace=False)
        
        self.list_patients = np.concatenate((self.list_patients_majority, 
                                             self.list_patients_minority))
       
        # Reduce the dataframe to only have the list of patients
        self.df = self.df[self.df.patient_id.isin(self.list_patients)]
        self.df.reset_index(inplace=True)


        
        # From the df, I will need to construct the list of paths for the desired images, the format will be patient_id/image_id.dcm
        self.list_images = []
        for patient in self.list_patients:
            for image in self.df.image_id[self.df.patient_id == patient].values:
                self.list_images.append(str(patient) + '/' + str(image) + '.jpg')
        self.list_images = sorted(self.list_images)
        self.labels = self.df.cancer.values
        self.labels = self.labels.astype(np.float32)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        path_image = self.path_patients + self.list_images[idx]
        image = plt.imread(path_image)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        # Convert the image to a 3 channel image
        image = image.unsqueeze(0)
        
        # image = image.permute(2, 0, 1)
        # Get the label it should be either 0 or 1
        label = self.labels[idx]
        image, label = image.cuda(), label.cuda()
        sample = {'image': image, 'label': label}
        if self.transform:
            image = self.transform(image)
            image = image/255
        return sample
    
class DatasetBreastUpsample(torch.utils.data.Dataset):
    def __init__(self, df, path_patients, view='CC', breast = 'L', transform=None):
        self.df = df
        self.path_patients = path_patients
        self.transform = transform
        self.view = view
        self.breast = breast
        # Working with the dataframe
        self.df = self.df[self.df.view == self.view]
        self.df = self.df[self.df.laterality == self.breast]
        # I need to randomly transform patients that are in the minority class
        self.list_patients_majority = np.unique(self.df.patient_id[(self.df.cancer == 0).values])
        self.list_patients_minority = np.unique(self.df.patient_id[(self.df.cancer == 1).values])
        # I will need to upsample the minority class
        self.list_patients_minority = np.random.choice(self.list_patients_minority,
                                                         size=len(self.list_patients_majority),
                                                            replace=True)
        self.list_patients = np.concatenate((self.list_patients_majority,
                                                self.list_patients_minority))
        # Reduce the dataframe to only have the list of patients
        self.df = self.df[self.df.patient_id.isin(self.list_patients)]
        self.df.reset_index(inplace=True)
        # From the df, I will need to construct the list of paths for the desired images, the format will be patient_id/image_id.dcm
        self.list_images = []
        
        for patient in self.list_patients:
            for image in self.df.image_id[self.df.patient_id == patient].values:
                self.list_images.append(str(patient) + '/' + str(image) + '.jpg')
        self.list_images = sorted(self.list_images)
        self.labels = self.df.cancer.values
        self.labels = self.labels.astype(np.float32)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        path_image = self.path_patients + self.list_images[idx]
        image = plt.imread(path_image)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        # Convert the image to a 3 channel image
        image = image.unsqueeze(0)
        
        # image = image.permute(2, 0, 1)
        # Get the label it should be either 0 or 1
        label = self.labels[idx]
        image, label = image.cuda(), label.cuda()
        if self.transform:
            image = self.transform(image)/255
        sample = {'image': image, 'label': label}

        return sample
