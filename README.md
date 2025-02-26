# Principled Positional Encodings for Medical Imaging

## Abstract
The adoption of Transformer-based architectures in the medical domain is growing rapidly. However, their suitability for medical tasks depends on architectural design choices, particularly the inductive biases they impose. In this study, we critically examine the role of Positional Encodings (PEs), arguing that commonly used approaches may be suboptimal for medical imaging. While Sinusoidal Positional Encodings (SPEs) and learned PEs have been widely adopted from Natural Language processing to computer vision, their capability to effectively preserve spatial structures in medical data has not been sufficiently studied. Fourier Positional Encodings (FPEs) have been proposed to better align with the spatial nature of images. However, in medical imaging conserved spatial structures like organs, high dimensionality, and anisotropy further complicate the demands on PEs.
To address these complications, we propose Anisotropic Fourier Feature Positional Encoding (AFPE), a generalization of FPE that is capable of incorporating anisotropic, class-specific and domain-specific spatial dependencies. We systematically benchmark AFPE against commonly used PEs on multi-label classification in chest x-rays and ejection fraction regression in echocardiography. Our results demonstrate that choosing the correct PE can significantly improve the models performance. We show that the optimal PE depends on the domain of interest and the anisotropy of the data. Finally, our proposed AFPE significantly outperforms state-of-the-art PEs in ejection fraction regression. We conclude that in anisotropic medical images and videos it is of paramount importance to choose an anisotropic positional encoding that fits the data.

## Installation
To run the code in this repo create a conda environment and install the dependencies using the following commands:
```bash
conda env create -f environment.yml # Creates conda environment
conda activate posenc # Activates the conda environment
python setup.py install # Installs the package
```

## Reproducing the results
To reproduce the final reported results and plots in the paper follow the notebooks in the `notebooks` folder. The notebooks are named according to the experiments they reproduce.

**Notebooks:**
- [ChestX](notebooks/ChestX.ipynb)
- [EchoNet-Dynamic](notebooks/EchoNet-Dynamic.ipynb)
- [Figures](notebooks/figures.ipynb)

This publication introduces Anisotropic Fourier Feature Positional Encodings (AFPE). The implementation of this positional encoding as well as all other positional encodings can be found in the [positional_encodings.py](posenc/nets/positional_encodings.py) file.

## Pretrained models
Pretrained models can be downloaded from [https://huggingface.co/afpe/afpe](https://huggingface.co/afpe/afpe).

## Training a model from scratch
To train a model like in the this publication you need to:
1. Download the datasets
2. Preprocess the datasets
3. Train the model

### 1. Datasets

The datasets used in this study are publicly available and can be accessed through the following links: 
- NIH Chest X-ray:  https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
- EchoNet-Dynamic:  https://echonet.github.io/dynamic/index.html

**Important:** After downloading the datasets change the *ROOT* variable in [posenc/datasets/chestx.py](posenc/datasets/chestx.py) and [posenc/datasets/echonet.py](posenc/datasets/echonet.py) to the path where the datasets are stored!

### 2. Preprocessing
ChestX preprocessing includes (see [script](posenc/datasets/chestx.py)):
- Resizing the images to 224x224
- Normalizing the images

EchoNet preprocessing includes (see [script](posenc/datasets/echonet.py)):
- Grayscale video
- Normalizing the videos

### 3. Training
To train any of the models in the paper, you can run the [train.py](posenc/train.py) script.
Use ```posenc/train.py --help``` to view all options.

Example to run the EchoNet-Dynamic regression task with the Anisotropic Fourier Positional Encoding (AFPE):
```bash
python posenc/train.py --task echonetreg --positional_encoding isofpe
```

Example to run the ChestX multi-label classification task with the Anisotropic Fourier Positional Encoding (AFPE):
```bash
python posenc/train.py --task chestxmulti --positional_encoding isofpe
```
