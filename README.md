# Alignment and Benchmarking for Drone Vision Dataset

## Abstract

This study deals with semantic segmentation of high-resolution drone imagery and the performance of deep convolutional neural networks (CNNs) on this task, by using automatically generated drone labels. As human annotation of the training data is a very expensive process, especially in the case of semantic segmentation where each pixel in the image is assigned a specific class, this paper addresses the feasibility of using generated drone labels instead of human annotated drone labels in the context of supervised semantic segmentation. By analysing a varied suite of algorithms for obtaining drone labels from already existing satellite labels, such as descriptor-based image matching, dense pixel correspondence, feature matching and warping, this work shows a promising direction in overcoming the time-consuming process of manually annotating the unmanned aerial vehicle (UAV) imagery with the purpose of reducing the time and costs of creating datasets required by various supervised deep learning tasks.

## 1.Data

The data that you download should be placed in a folder, named for example ```drone_dataset```, under the PROJECT_ROOT directory,
at the same level with the ```labeling``` folder. After that, you must follow the ```labeling``` [README](labeling/README.md)
to understand the data better, learn how to generate satellite images and labels and re-organise the directory structure in a suitable format for further experiments.

## 2.Project structure and usage

The entire project should also be uploaded do Google Drive under a PROJECT_ROOT directory, named for example ```SemesterProject```, as some parts of the code can only be run in a Google Colab environment.

As aforementioned in the previous section, the ```labeling``` folder contains data creation and organisation instruction and scripts and can be run locally.

The ```PatchMatch``` folder contains the PatchMatch algorithm and can be run locally. More instructions can be found at [README](PatchMatch/README.md).

The ```SCOT``` folder contains the code for SCOT algorithm. This algorithm must be run in a Google Colab environemnt. The entire code can be found in this [SCOT notebook](SCOT/SCOT.ipynb), containing further explanations, usage instructions and environment setup.

The ```TPS_LoFTR``` folder contains the code for LoFTR and TPS and must be run in a Google Colab environment. The entire code can be found in this [TPS_LoFTR notebook](TPS_LoFTR/TPS_LoFTR.ipynb), containing further explanations, usage instructions and environment setup.


