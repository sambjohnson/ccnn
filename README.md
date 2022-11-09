# ccnn

Cortical CNN's

This project uses a 2D convolutional neural network to segment a cortical surface into different anatomical regions.

The primary region of interest is the Occipitotemporal Sulcus (OTS). The network achieves this using two strategies.

- Pretraining / finetuning: the CNN is first trained on the collection of anatomical labels that constitute the Destrieux atlas, as automatically labeled by the program Freesurfer. The dataset for these labels is the thousands of subjects in the Healthy Brain Network (HBN) dataset. Following this initial pretraining stage, the network is finetuned on a small collection of human-created labels.
- 2d training -> 3d prediction: while the network itself operates on 2d images (views of the cortex from specific angles), the curved surface of the cortex is the object of interest; therefore, the predictions are finally translated back to this surface, where anatomical labels as predicted from different 2d views may be combined using various ensembling logics.

The architecture is a 2d UNet with a Resnet-18 backbone, trained via the Adam optimizer.

Project:
- The primary utilities for the UNet architecture and training can be found in the directory UNet. In addition, this directory also contains the files pipeline_utilities.py and pipeline scripts, which can be run to generate 2d images for network training.
- Auxilary utilities to streamline PyTorch are located in the directory dlutil.
- To train and evaluate a model interactively, use the notebook unet-transfer.ipynb
