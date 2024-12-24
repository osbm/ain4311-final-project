import os
import torch
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import torchvision
from torchvision import models, transforms
from tqdm import tqdm
# import multiclass f1 score and multiclass confusion matrix

from sklearn.metrics import f1_score, confusion_matrix
from torchvision import datasets

import random
import uuid


male_path = "/home/osbm/Documents/git/ain4311-final-project/data/man"
female_path = "/home/osbm/Documents/git/ain4311-final-project/data/woman"
cat_path = "/home/osbm/Documents/git/ain4311-final-project/data/Cat"
dog_path = "/home/osbm/Documents/git/ain4311-final-project/data/Dog"
rng = random.Random(42)


for class_name, class_folder in zip(["male", "female", "cat", "dog"], [male_path, female_path, cat_path, dog_path]):
    images = os.listdir(class_folder)
    images = [os.path.join(class_folder, img) for img in images]
    rng.shuffle(images)
    num_images = len(images)
    train_images = images[:8000]
    val_images = images[8000:9300]
    for split, images in zip(["train", "val"], [train_images, val_images]):
        if os.path.exists(f"data/{split}/{class_name}"):
            continue
        os.makedirs(f"data/{split}/{class_name}", exist_ok=False)
        for image in tqdm(images, desc=f"Copying {split} images for {class_name}"):
            # give the images a unique name
            image_extension = image.split(".")[-1]
            image_folder = ''.join(image.split("/")[-2:])
            image_name = f"{image.split('/')[-1]}.{image_extension}"
            new_image_name = f"{uuid.uuid4()}.{image_extension}"
            # original image path may contain spaces, so we need to wrap it in quotes
            os.system(f"cp '{image}' 'data/{split}/{class_name}/{new_image_name}'")


val_transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder("data/train", transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder("data/val", transform=val_transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# create the model

device = torch.device("cuda")



class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 15 * 15, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)

model = BasicCNN(num_classes=4)
model = model.to(device)


model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


# print number of parameters and amount of memory needed according to weight dtypes
num_params = sum(p.numel() for p in model.parameters())
memory = sum(p.numel() * p.element_size() for p in model.parameters())

print(f"Model has {num_params} parameters")
print(f"Model uses {memory/1024**2:.2f} MB of memory")


history = {
    "train-loss": [],
    "val-loss": [],
    "f1": []
}
num_epocs = 100

best_f1 = 0
for epoch_idx in range(num_epocs):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    history["train-loss"].append(train_loss)

    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    for images, labels in tqdm(val_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        val_loss += loss.item()
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    history["val-loss"].append(val_loss)

    f1 = f1_score(all_labels, all_preds, average="weighted")
    if f1 > best_f1:
        best_f1 = f1
        print(f"Better model found with f1: {f1}")
        torch.save(model.state_dict(), "best-model.pth")
    # confusion matrix just prints the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    history["f1"].append(f1)

    print(f"Epoch {epoch_idx+1}/{num_epocs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} F1: {f1:.4f}")
    print(cm)

    history_df = pd.DataFrame(history)
    history_df.plot()
    plt.savefig("pretraining-history.png")
    history_df.to_csv("history.csv", index=False)

