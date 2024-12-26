import streamlit as st
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import io

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, str(label.item()), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# Function to convert image to bytes for download
def convert_image_to_bytes(image):
    img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Streamlit app title
st.title("Automatic Bounding Box Generator")

# Option to upload an image or use the webcam
option = st.selectbox("Select Input Method", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Convert image to tensor
        image_tensor = F.to_tensor(image).unsqueeze(0)

        # Perform object detection
        with torch.no_grad():
            predictions = model(image_tensor)

        # Get bounding boxes and labels
        boxes = predictions[0]['boxes'].numpy()
        scores = predictions[0]['scores'].numpy()
        labels = predictions[0]['labels'].numpy()

        # Filter out weak detections
        threshold = 0.5
        strong_boxes = boxes[scores > threshold]
        strong_labels = labels[scores > threshold]

        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image_np.copy(), strong_boxes, strong_labels)
        st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

        # Convert to bytes for download
        img_bytes = convert_image_to_bytes(image_with_boxes)
        st.download_button(
            label="Download Image with Bounding Boxes",
            data=img_bytes,
            file_name="image_with_bounding_boxes.png",
            mime="image/png",
            help="Click to download the processed image."
        )

elif option == "Use Webcam":
    # Camera input widget
    picture = st.camera_input("Take a picture")

    if picture:
        # Convert the UploadedFile to a PIL Image
        image = Image.open(picture)
        image_np = np.array(image)

        # Convert image to tensor
        image_tensor = F.to_tensor(image).unsqueeze(0)

        # Perform object detection
        with torch.no_grad():
            predictions = model(image_tensor)

        # Get bounding boxes and labels
        boxes = predictions[0]['boxes'].numpy()
        scores = predictions[0]['scores'].numpy()
        labels = predictions[0]['labels'].numpy()

        # Filter out weak detections
        threshold = 0.5
        strong_boxes = boxes[scores > threshold]
        strong_labels = labels[scores > threshold]

        # Draw bounding boxes
        image_with_boxes = draw_bounding_boxes(image_np.copy(), strong_boxes, strong_labels)
        st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

        # Convert to bytes for download
        img_bytes = convert_image_to_bytes(image_with_boxes)
        st.download_button(
            label="Download Image with Bounding Boxes",
            data=img_bytes,
            file_name="image_with_bounding_boxes.png",
            mime="image/png",
            help="Click to download the processed image."
        )

# Additional mobile-friendly enhancements
st.markdown("""
### Instructions:
- **Upload Image**: Select an image from your device to generate bounding boxes.
- **Use Webcam**: Capture a photo directly using your device's camera.
- **Download**: After processing, you can download the image with bounding boxes.
""")