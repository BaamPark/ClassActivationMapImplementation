# Class Activation Maps (CAM) Visualization for PyTorch ResNet Model

This repository contains a script that enables you to visualize the class activation maps (CAM) of a trained PyTorch ResNet model. CAM helps in understanding which regions of an input image contribute most to the classification decision made by the model. The provided script takes a dataset of images and generates a heatmap overlay on each image, highlighting the regions that are most influential for the predicted class.

## Requirements
- Python 3.x
- numpy
- cv2 (OpenCV)
- torch
- torchvision
- tqdm
- PIL (Pillow)

## Usage
1. Train your PyTorch ResNet model on a dataset.
2. Set your model to evaluation mode after training.
3. Uncomment the display_cam function call in the main() function from cam.py
4. Ensure that the getitem method of your PyTorch dataset returns a numpy array image and that the label tensor is one-hot encoded.
5. Run the script to generate class activation maps.

## Functions 
`getCAM()`
This function computes the class activation map for the target class using the feature maps from the last convolutional layer and the weights of the fully connected layer.


`cam_pass_input()`
This function computes the class activation map for the target class using the feature maps from the last convolutional layer and the weights of the fully connected layer.

`display_cam()`
This function generates CAM visualizations for a given dataset of test images. It applies the heatmap overlay to each image and saves the resulting visualization.

## Refernce
- http://www.snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html.html
