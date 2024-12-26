# Automatic Bounding Box Generator

## Overview

The **Automatic Bounding Box Generator** is a web application built using Streamlit that allows users to upload images or capture photos using their webcam. The application utilizes a pre-trained Faster R-CNN model to automatically detect significant objects in the images and draw bounding boxes around them. Users can then download the processed images with the bounding boxes.

## Features

- **Image Upload**: Users can upload images in JPG, JPEG, or PNG formats.
- **Webcam Capture**: Users can take pictures directly from their webcam.
- **Automatic Object Detection**: The app uses a pre-trained Faster R-CNN model to detect objects in the images.
- **Bounding Box Visualization**: Detected objects are highlighted with bounding boxes and labels.
- **Download Processed Images**: Users can download the images with bounding boxes for further use.

## Requirements

To run this application, you need to have the following libraries installed:

- Python 3.x
- Streamlit
- OpenCV
- NumPy
- PyTorch
- torchvision
- Pillow

You can install the required libraries using pip:

```bash
pip install streamlit opencv-python numpy torch torchvision Pillow
```
