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

from model import BasicCNN

calibration_transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor()
])

calibration_dataset = torchvision.datasets.ImageFolder("data/finetuning_train", transform=calibration_transform)

def run_calibration(model, calibration_dataset):

    # Set the model to the evaluation
    model.eval()

    for image, _ in tqdm(calibration_dataset):
        image = image.unsqueeze(0)
        model(image)





def print_size_of_model(model):
    # print number of parameters and number of bytes

    num_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print('Number of parameters: %d' % num_params)
    print('Size of the model: %d bytes' % total_size)

def main():
    model = BasicCNN(num_classes=4)
    model.load_state_dict(torch.load('best-finetuning-model.pth'))
    model.eval()
    print_size_of_model(model)

    # Quantize the model by just
    quantized_model = torch.quantization.quantize(
        model, run_fn=run_calibration, run_args=[calibration_dataset]
    )
    print_size_of_model(quantized_model)
    print(quantized_model)


    # Check the accuracy of the quantized model
    # Set the model to the evaluation mode
    quantized_model.eval()


    # i will need to export this model as a cpp file

    # print first layer of the model to see the quantization parameters
    params = list(quantized_model.parameters())


    for i in range(len(params)):
        print(params[i].shape, params[i].dtype)

    print("Parameters:")
    for name, param in quantized_model.named_parameters():
        print(f"{name}: {param.size()} {param.dtype}")


        # Print comparison
    print("Original model parameters:")
    for param in model.parameters():
        print(param.size(), param.dtype)

    print("\nQuantized model parameters:")
    for param in quantized_model.parameters():
        print(param.size(), param.dtype)


    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.quantized.dynamic.Linear):
            print(f"Layer: {name}")
            print("Quantized weights:", module.weight())
            print("Weight dtype:", module.weight().dtype)

    # export model to onnx

    dummy_input = torch.randn(1, 3, 120, 120)
    torch.onnx.export(quantized_model, dummy_input, "best-finetuning-model-quantized.onnx")

    # Save the quantized model
    torch.save(quantized_model.state_dict(), 'best-finetuning-model-quantized.pth')


if __name__ == '__main__':
    main()