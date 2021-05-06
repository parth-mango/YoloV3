# Capstone Project 

## Problem Statement 

### The task here was to work on safety equipment dataset, namely - Hardhat, Boots, Mask and Vest. The challenge was to generate bounding boxes using YoloV3 archietecture, masks/planer images using Planercnn archietecture and depth images using Midas Architecture on the given images in the dataset. From the model point of view, all the three algorithms are based on encoder decoder archietecture thus the challenging part was to combine the three models together to produce satisfactory results. 

## My Solution

### Model Archiecteture
We use the midas encoder - Resnext 101 as the common encoder and then attach the yolo decoder, midas decoder and planercnn decoder to it. This is a multihead model with different types of output in different heads. The losses are combined as "total loss" which is backpropogated through the yolo and planercnn decoder to train the decoders while the common encoder and midas decoder are used with backpropagation. 

1. While attaching the resnext encoder and yolo decoder we pinpoint layer 36, 61 and 75 which are connected to layers of encoder layers namely 256, 512 and 1024. We use 1x1 kernels and connect it with the darknet 53 model replacing the yolo encoder. 
2. For Planercnn model we connect the resnext encoder with the FPN branch replacing the resnet encoder. Here we also rename the predict function of planer decoder file to forward function.

### Supportive Changes:
1. The config file of planer cnn was changed to turn off the depth map prediction as we use the depth map from the midas branch.
2. A make_date.py file was written that combines the loading function of different archietecture. Help from several sources were taken.
3. Argument parsers were combined and written off in parser.py file to ease passing of the arguments.
4. In train.py file the training functions of planercnn and yolo were included to train the layers of respective deocders
5. The evaluate code was used from Midas intel repository to give depth map predictions
6. The loss of each of the three heads were combined and was backpropagated into planercnn and yolo decoder.
7. SSIM loss was also calculated beside rmse loss to further improve the performance of the model.


## Current Development

The model seems to be working as a combined multi headed archietecture. The training started but upon running it gives a notorious error where the total loss gives "nan" value. Upon delving into the error this is arising from the issue in class labels. I have been trying to find a solution but the error seems to persist. 


## Ending Remarks

The error can be solved the error if i had some more time to work on it. I would like to submit my current progress as my final submission to the capstone project. It was ending a very challenging and interesting project. Thanks for giving me this opportunity.


 
