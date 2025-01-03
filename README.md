# Our Codebase for AIN4311 project


## Project Description



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


### Arduino Codes

There is two different ardunio photo taker code bases for our project.

#### Telegram photo taker



#### ESP32-CAM web server photo taker


#### Inference on the ESP32-CAM
