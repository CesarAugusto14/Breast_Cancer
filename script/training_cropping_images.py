import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from torchsummary import summary
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Custom modules:
from dataset_creation import DatasetBreastDownsample
from train_scripts import train

torch.manual_seed(1311)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv('./data/train.csv')
# df.sort_values(by='patient_id', inplace=True)
path_images = './data/256_images_cropped/'
# list_images = sorted(os.listdir(path_images))
modality = 'CC'
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
train_size = int(0.6 * len(dset_trial))
val_size = int(0.2 * len(dset_trial))
test_size = len(dset_trial) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dset_trial, [train_size, val_size, test_size])


# Create the dataloaders
size_batch = 16
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
# model = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels = 8, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(in_channels=8, out_channels = 16, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(in_channels=16, out_channels = 32, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(in_channels=32, out_channels = 64, kernel_size=3, stride=1, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(64*16*16, 100),
#     nn.ReLU(),
#     nn.Linear(100, 2),
#     nn.ReLU(),
#     nn.Linear(2, 1),
#     nn.Sigmoid()
# )
model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size = 2, stride = 2))
model.add_module('conv2', nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size = 2, stride = 2))
model.add_module('conv3', nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size = 2, stride = 2))
# flatten
model.add_module('flatten', nn.Flatten())
model.add_module('fc1', nn.Linear(in_features = 32768, out_features = 512))
model.add_module('relu4', nn.ReLU())
model.add_module('dropout', nn.Dropout(p = 0.5))
model.add_module('fc2', nn.Linear(in_features = 512, out_features = 256))
model.add_module('relu5', nn.ReLU())
model.add_module('dropout', nn.Dropout(p = 0.5))
model.add_module('fc3', nn.Linear(in_features = 256, out_features = 1))
model.add_module('sigmoid', nn.Sigmoid())

if torch.cuda.is_available():
    model.cuda()
# print(model)
summary(model, (1, 256, 256))

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer with l2 regularization
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1, momentum=0.9)

num_epochs = 20

model, loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid = train(model, num_epochs, train_loader, val_loader, track = True, 
                                                                                          optimizer = optimizer, criterion= criterion)

plt.figure(figsize=(10, 10))
plt.plot(loss_hist_train, label='Training loss')
plt.plot(loss_hist_valid, label='Validation loss')
plt.legend(frameon=False)
plt.title("Loss")
plt.savefig('./results/loss.png')

plt.figure(figsize=(10, 10))
plt.plot(accuracy_hist_train, label='Training accuracy')
plt.plot(accuracy_hist_valid, label='Validation accuracy')
plt.legend(frameon=False)
plt.title("Accuracy")
plt.savefig('./results/accuracy.png')
# test set
model.eval()
with torch.no_grad():
    loss_hist_test = 0
    accuracy_hist_test = 0

    for i, sample_batched in enumerate(test_loader):
        outputs = model(sample_batched['image'])
        loss = criterion(outputs, sample_batched['label'].unsqueeze(1))
        loss_hist_test += loss.item()
        accuracy_hist_test += (outputs.round() == sample_batched['label'].unsqueeze(1)).sum().item()
    loss_hist_test /= len(test_loader)
    accuracy_hist_test /= len(test_loader.dataset)

    print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(loss_hist_test, accuracy_hist_test*100))

# Compute the ROC curve
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for i, sample_batched in enumerate(test_loader):
        outputs = model(sample_batched['image'])
        y_pred.extend(outputs.tolist())
        y_true.extend(sample_batched['label'].tolist())
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print("AUC: {:.2f}".format(auc))
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label = f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig("./results/roc_cropped.png")

with torch.no_grad():
    y_pred = []
    y_true = []
    for i, sample_batched in enumerate(test_loader):
        outputs = model(sample_batched['image'])
        y_pred.extend(outputs.round().tolist())
        y_true.extend(sample_batched['label'].tolist())
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./results/confusion_cropped.png")