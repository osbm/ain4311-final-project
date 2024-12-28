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

train_dataset = torchvision.datasets.ImageFolder("data/finetuning_train", transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder("data/finetuning_val", transform=val_transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# create the model

device = torch.device("cuda")

from model import BasicCNN

model = BasicCNN(num_classes=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


# print number of parameters and amount of memory needed according to weight dtypes
num_params = sum(p.numel() for p in model.parameters())
memory = sum(p.numel() * p.element_size() for p in model.parameters())

print(f"Model has {num_params} parameters")
print(f"Model uses {memory/1024**2:.2f} MB of memory")

model.load_state_dict(torch.load("best-pretraining-model.pth"))


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
        torch.save(model.state_dict(), "best-finetuning-model.pth")
    # confusion matrix just prints the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    history["f1"].append(f1)

    print(f"Epoch {epoch_idx+1}/{num_epocs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} F1: {f1:.4f}")
    print(cm)

    history_df = pd.DataFrame(history)
    history_df.plot()
    plt.savefig("finetuning-history.png")
    history_df.to_csv("finetuning-history.csv", index=False)

