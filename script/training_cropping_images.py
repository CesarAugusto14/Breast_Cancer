import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from torchsummary import summary

# Custom modules:
from dataset_creation import DatasetBreastDownsample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv('./data/train.csv')
# df.sort_values(by='patient_id', inplace=True)
path_images = './data/256_images_cropped/'
# list_images = sorted(os.listdir(path_images))
modality = 'MLO'
which_breast = 'L'
dset_trial = DatasetBreastDownsample(df, path_images, view=modality, breast=which_breast)

## Test the DatasetBreastDownsample class, check if the images are loaded correctly and if the labels are correct:

# sample = dset_trial[0]
# sample2 = dset_trial[1]
# print('Sample 1:')
# print(sample['image'].shape)
# print(sample['label'])
# print('Sample 2:')
# print(sample2['image'].shape)
# print(sample2['label'])

# print(dset_trial.df.head(10))
# # Show the image as a numpy array int32
# plt.figure(figsize=(20, 10))
# plt.subplot(1,2,1)
# plt.imshow(sample['image'].numpy().astype(np.int32), cmap='gray')
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(sample2['image'].numpy().astype(np.int32), cmap='gray')
# plt.axis('off')
# plt.show()

# Now that we know that this work, let's create the dataloaders:

# Train-validation-test split
train_size = int(0.8 * len(dset_trial))
val_size = int(0.1 * len(dset_trial))
test_size = len(dset_trial) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dset_trial, [train_size, val_size, test_size])


# Create the dataloaders
size_batch = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=size_batch, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=size_batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=size_batch, shuffle=True)

# Check if it is working
# for i_batch, sample_batched in enumerate(train_loader):
#     print(i_batch, sample_batched['image'].size(), sample_batched['label'])
#     if i_batch == 3:
#         break

# Now that we have the dataloaders, we can create the model and train it using nn.Sequential:

# Create the model
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels = 8, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=8, out_channels = 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels = 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels = 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64*16*16, 100),
    nn.ReLU(),
    nn.Linear(100, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)
if torch.cuda.is_available():
    model.cuda()
# print(model)
summary(model, (1, 256, 256))

# Define the loss function
criterion = nn.BCELoss()
# Define a loss function with l2 regularization
# criterion = nn.BCELoss() + 0.01*torch.norm(model.fc1.weight, 2) + 0.01*torch.norm(model.fc2.weight, 2)

# Define the optimizer with l2 regularization
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1, momentum=0.5)