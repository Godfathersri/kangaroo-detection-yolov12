# Kangaroo-detection-yolov12

This project fine-tunes a YOLOv12 object detection model to detect kangaroos in images and videos. The goal of this project was to build a lighweited and accurate wildlife detection system that could support real-world use cases such as road monitoring and wildlife collision prevention.

## Problem

Kangroos are a major contributer to animal-related vehicle collisions in Australia. Exitsting general object detection models are often not optimized for Australian widlife, so project focuses on training a custom model dataset specifically for kangroo detection. 

## Tech Stack

-Python
-PyTorch
-YOLOv12
-Ultralytics
-Roboflow
-OpenCV

## Dataset

The model was trained using 500+ annotated kangaroo images. Images were collected from open source platforms and manually annotated using Roboflow. 

Due to dataset size, this repository include only a small sample dataset to show the expected format. 

Data augmentation techniques were applied to improve model generalization, including: 

-Flipping
-Rotating
-Brightness and contrast adjustment
-Mosaic augmentation

## Training Apprach 

The training workflow included:

1. Preparing and annotating the dataset 
2. Splitting data into training and validation sets
3. Training a baseline YOLOv12 model 
4. Applying data augmentation 
5. Fine-tuning the model 
6. Evaluating model performance

## Results

The model improved significatly after augmentation and iterative training

| Model | Epochs | Augmentation | F1 Score | Precision | Recall | mAP@0.5 |
|---|---:|---|---:|---:|---:|---:|
| Baseline | 40 | No | 0.66 | 1.00 | 0.91 | 0.65 |
| Extended | 100 | No | 0.73 | 1.00 | 0.85 | 0.71 |
| Final | 100 | Yes | 0.99 | 1.00 | 0.99 | 0.99 |

## How to Run 

pip install -r requirment.txt

# Train the model 

python scripts/train_val_pre.py --mode train --data data_sample/data.yaml

## Validate the model 

python scripts/train_val_pre --mode val --weights runs/train/weights/best.pt --data data_sample/data.yaml

## Run Prediction 

python scripts/train_val_pre.py --mode predict --weights runs/train/weights/best.pt --predict_source data_sample/test/images

## Documentation 

A detailed project overview is avalible here:
[Project Overview (PDF)](ICT_Innovation.pdf)