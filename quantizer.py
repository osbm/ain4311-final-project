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




# Calibration function to compute min and max for activations
def calibrate_model(model, data_loader):
    layer_stats = {}
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            x = data
            for name, layer in model.named_children():
                x = layer(x)
                if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Quantize these layers
                    layer_stats[name] = {
                        'min': x.min(),
                        'max': x.max()
                    }
    return layer_stats

# Helper function to quantize a tensor to int8
def quantize_tensor(tensor, scale, bitwidth=8):
    quantized = torch.round(tensor / scale).clamp(-2**(bitwidth - 1), 2**(bitwidth - 1) - 1)
    return quantized.to(torch.int8)

# Quantization loop using calibration dataset
def quantize_model(model, data_loader):
    # First, gather statistics using the calibration dataset
    layer_stats = calibrate_model(model, data_loader)

    # Quantize model layers using the calibration statistics
    for name, param in model.named_parameters():
        if isinstance(param, torch.nn.Parameter):
            # Match the layer name to the stats and apply quantization
            for layer_name, stats in layer_stats.items():
                if layer_name in name:  # Match layer name
                    # Compute scale for quantization (based on max value)
                    scale = stats['max'] / (2**7 - 1)

                    # Quantize the weights of the parameter
                    quantized_weights = quantize_tensor(param.data, scale)

                    # Update the parameter with the quantized weights
                    param.data.copy_(quantized_weights)  # In-place update using copy_

                    # Optionally, store the scale for later dequantization (during inference)
                    param.scale = scale

    return model

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
    data_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=64, shuffle=True)
    quantized_model = quantize_model(model, data_loader)



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



    # save all the weights and biases of the model
    # in a text file as a list of numbers

    with open("best-finetuning-model-quantized.txt", "w") as f:
        for name, param in quantized_model.named_parameters():
            f.write(f"{name}\n")
            f.write(f"{param.size()}\n")
            f.write(f"{param.flatten().tolist()}\n")



    # Save the quantized model
    torch.save(quantized_model.state_dict(), 'best-finetuning-model-quantized.pth')


if __name__ == '__main__':
    main()