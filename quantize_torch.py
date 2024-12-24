import os
import subprocess
from typing import Iterable, List, Tuple

import torch
import torchvision

from ppq import QuantizationSettingFactory, QuantizationSetting
from ppq.api import espdl_quantize_torch, get_target_platform
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torch.nn as nn
import urllib.request
import zipfile


def convert_relu6_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_relu6_to_relu(child)
    return model


def quant_setting_mobilenet_v2(model: nn.Module, optim_quant_method: List[str] = None,) -> Tuple[QuantizationSetting, nn.Module]:
    """Quantize torch model with optim_quant_method.

    Args:
        optim_quant_method (List[str]): support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'
        -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispathching
                                        the layers to 16-bit quantization. You can remove or add layers according to your
                                        needs.
        -'LayerwiseEqualization_quantization'： using weight equalization strategy, which is proposed by Markus Nagel.
                                                Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                Since ReLU6 exists in MobilenetV2, convert ReLU6 to ReLU for better precision.

    Returns:
        [tuple]: [QuantizationSetting, nn.Module]
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    if optim_quant_method is not None:
        if "MixedPrecision_quantization" in optim_quant_method:
            # These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
            # You can remove or add layers according to your needs.
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.0/Conv",
                get_target_platform(TARGET, 16),
            )
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.2/Clip",
                get_target_platform(TARGET, 16),
            )
        elif "LayerwiseEqualization_quantization" in optim_quant_method:
            # layerwise equalization
            quant_setting.equalization = True
            quant_setting.equalization_setting.iterations = 4
            quant_setting.equalization_setting.value_threshold = 0.4
            quant_setting.equalization_setting.opt_level = 2
            quant_setting.equalization_setting.interested_layers = None
            # replace ReLU6 with ReLU
            model = convert_relu6_to_relu(model)
        else:
            raise ValueError(
                "Please set optim_quant_method correctly. Support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'"
            )

    return quant_setting, model

def quant_setting_basic_cnn(model: nn.Module, optim_quant_method: List[str] = None,) -> Tuple[QuantizationSetting, nn.Module]:
    """Quantize torch model with optim_quant_method.

    Args:
        optim_quant_method (List[str]): support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'
        -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispathching
                                        the layers to 16-bit quantization. You can remove or add layers according to your
                                        needs.
        -'LayerwiseEqualization_quantization'： using weight equalization strategy, which is proposed by Markus Nagel.
                                                Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                Since ReLU6 exists in MobilenetV2, convert ReLU6 to ReLU for better precision.

    Returns:
        [tuple]: [QuantizationSetting, nn.Module]
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    if optim_quant_method is not None:
        if "MixedPrecision_quantization" in optim_quant_method:
            # These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
            # You can remove or add layers according to your needs.
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.0/Conv",
                get_target_platform(TARGET, 16),
            )
            quant_setting.dispatching_table.append(
                "/features/features.1/conv/conv.0/conv.0.2/Clip",
                get_target_platform(TARGET, 16),
            )
        elif "LayerwiseEqualization_quantization" in optim_quant_method:
            # layerwise equalization
            quant_setting.equalization = True
            quant_setting.equalization_setting.iterations = 4
            quant_setting.equalization_setting.value_threshold = 0.4
            quant_setting.equalization_setting.opt_level = 2
            quant_setting.equalization_setting.interested_layers = None
            # replace ReLU6 with ReLU
            model = convert_relu6_to_relu(model)
        else:
            raise ValueError(
                "Please set optim_quant_method correctly. Support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'"
            )

    return quant_setting, model


def collate_fn1(x: Tuple) -> torch.Tensor:
    return torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0)


def collate_fn2(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


if __name__ == "__main__":
    BATCH_SIZE = 32
    INPUT_SHAPE = [3, 120, 120]
    DEVICE = "cuda"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
    TARGET = "esp32s"
    NUM_OF_BITS = 8
    ESPDL_MODEL_PATH = "quantized_custom.espdl"
    CALIB_DIR = "./imagenet"

    from model import BasicCNN

    model = BasicCNN(num_classes=4)

    # model = torchvision.models.mobilenet.mobilenet_v2(
    #     weights=MobileNet_V2_Weights.IMAGENET1K_V1
    # )
    model.load_state_dict(torch.load("best-model.pth"))
    model = model.to(DEVICE)



    # -------------------------------------------
    # Prepare Calibration Dataset
    # --------------------------------------------

    train_dataset = torchvision.datasets.ImageFolder("data/train", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -------------------------------------------
    # Quantize Torch Model.
    # --------------------------------------------

    # create a setting for quantizing your network with ESPDL.
    # if you don't need to optimize quantization, set the input 1 of the quant_setting_mobilenet_v2 function None
    # Example: Using LayerwiseEqualization_quantization
    # quant_setting, model = quant_setting_mobilenet_v2(
    #     model, ["LayerwiseEqualization_quantization"]
    # )

    quant_setting, model = quant_setting_basic_cnn(
        model, ["LayerwiseEqualization_quantization"]
    )


    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target="esp32s",
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn2,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
    )

    # -------------------------------------------
    # Evaluate Quantized Model.
    # --------------------------------------------
    # TODO