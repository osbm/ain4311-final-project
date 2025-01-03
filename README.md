# Our Codebase for AIN4311 project


## Project Description

This study presents a comprehensive approach to developing an AI-driven image classification system on the ESP32-CAM platform. The project leverages the device's small form factor, integrated camera, and Wi-Fi connectivity to enable real-time image acquisition and processing. Using lightweight convolutional neural networks (CNNs), the system is optimized through techniques such as model quantization, pruning, and transfer learning to meet the computational and memory limitations of the ESP32-CAM. The methodology includes preprocessing data to match hardware specifications, deploying TensorFlow Lite models, and utilizing efficient training workflows to adapt to the device's constraints. The outcomes demonstrate a balance between high efficiency and cost-effectiveness, emphasizing advantages such as low power consumption, ease of integration, and suitability for IoT applications. Despite challenges like camera quality, CPU clock speed, and heat dissipation, the project highlights the potential of the ESP32-CAM in deploying AI at the edge, offering insights into practical implementations in surveillance, environmental monitoring, and smart devices.

## Installation

### Prerequisites

#### Nix Package Manager
Our codes were primarily developed on a nix development environment. All necessary tooling is described inside the `flake.nix` file. To build the development environment, all you need to do is to run the following command in the project root directory:

```bash
$ nix develop
```


#### Standart Linux Environment

If you are not using nix, you can still run the project by installing the following dependencies, but we do not guarantee that it will work as expected.

- Python 3.12
- pip
- and the packages listed in the `requirements.txt` file.

To install the dependencies, you can run the following command:

```bash
$ pip install -r requirements.txt
```

You will need to install the arduino-ide package from your package manager. For example, on Ubuntu, you can run the following command:

```bash
$ sudo apt install arduino
```



### Preparing datasets

First you will need to prepare the datasets. You can download [this human faces dataset](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) and [this dogs and cats dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).

After fetching the datasets, you can run the following command to prepare the datasets:

```bash
$ cd data/
$ python seperate-pretraining-dataset.py
```

Also After placing the our custom dataset in the `data/` directory, you can run the following command to prepare the dataset:

```bash
$ cd data/
$ python create-finetuning-dataset.py
```

Now these scripts also speperate the dataset into training and validation sets. You can find the prepared datasets in the `data/` directory. The main dataset is the finetuning dataset that we took with out ESP32-CAM camera.


### Training the models

We packed our entire process into 3 python scripts. To get the initial pretraining model for the transfer learning process, you can run the following command:

```bash
$ python pretrainer.py
```


To get the final model, you can run the following command:

```bash
$ python finetuning.py
```


To get the quantization code, you can run the following command:

```bash
$ python quantizer.py
```

After that, you need to convert the quantized model to a tflite model. You can run the following command:

```bash
$ python onnx-to-tf.py
$ python tf-to-tflite.py
```

Now that we have the tflite model, we need  to convert it to a C header file. You can run the following command:

```bash
$ xxd -i model.tflite > model.h
```


### Arduino Codes

There is two main photo taker algorithms we used.

#### Telegram photo taker

You can find in `arduino/photo-collecter` directory. Keep in mind that you will need to edit this code with your own telegram bot token and chat id and also the wifi credentials.

Required libraries:
- ArduinoJson
- UniversalTelegramBot
- esp32-camera

#### Web server photo taker

You can find the code for this in the `arduino/web-server` directory. Keep in mind that you will need to edit this code with your own wifi credentials and also telegram bot token and chat id. We needed telegram because when we used android mobile hotspot there is no way to get the ip address of the ESP32-CAM. So this script sends the ip address to the telegram bot.

Required libraries:
- esp32-camera
- WiFi
- ArduinoJson
- UniversalTelegramBot

#### Inference on the ESP32-CAM

The code is inside the `arduino/inference` directory. You can find the code for the inference on the ESP32-CAM. You will also need wifi and telegram credentials but also you will need to add the model.h file to this folder.

Required libraries:
- esp32-camera
- WiFi
- ArduinoJson
- UniversalTelegramBot
- Adafruit TensorFlowLite