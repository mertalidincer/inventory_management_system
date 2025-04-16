from ultralytics import YOLO
import numpy as np
import cv2
import os

def get_model():  # Prepare the YOLO model
    model = YOLO('models/last (2).pt')  # Load the YOLO model from the specified path
    model.fuse()  # Fuse the model for faster inference
    return model

def plot_bboxes(results):
    """
    Draw bounding boxes on the image based on the detection results.
    
    Args:
        results: Detection results from the YOLO model.
        img: Original image to draw bounding boxes on.
    
    Returns:
        Image with bounding boxes drawn.
    """
    img = results[0].orig_img  # Get the original image
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32)  # Extract bounding boxes as integer coordinates
    num_objects = len(boxes)  # Count the number of detected objects
    print(f"Number of detected products: {num_objects}")  # Print the detection count
    for bbox in boxes:  # Loop over all bounding boxes
        img = cv2.rectangle(img, (bbox[0], bbox[1]),  # Draw a rectangle for each bounding box
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),  # Red color for the bounding box
                            thickness=1)  # Thickness of the rectangle
    return img

# Run the model on the input image
results = get_model()('MRI Competitors/Danube nahda (7).jpeg')
img = plot_bboxes(results)  # Plot bounding boxes on the image
cv2.imshow('img', img)  # Display the image with bounding boxes
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows


"""
    ------
"""

"""
This script processes images in a folder, detects objects using a YOLO model, 
draws bounding boxes on the detected objects, and saves the labeled images 
to an output folder.
"""


def get_model():  
    """
    Load and prepare the YOLO model for inference.
    """
    model = YOLO('models/best.pt')  # Load the YOLO model from the specified path
    model.fuse()  # Fuse the model for faster inference
    return model

def plot_bboxes(results, img):  
    """
    Draw bounding boxes on the image based on the detection results.
    
    Args:
        results: Detection results from the YOLO model.
        img: Original image to draw bounding boxes on.
    
    Returns:
        Image with bounding boxes drawn.
    """
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32)  # Extract bounding boxes as integer coordinates
    num_objects = len(boxes)  # Count the number of detected objects
    print(f"Number of detected products: {num_objects}")  # Print the detection count
    for bbox in boxes:  # Loop over all bounding boxes
        img = cv2.rectangle(img, (bbox[0], bbox[1]),  # Draw a rectangle for each bounding box
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),  # Red color for the bounding box
                            thickness=1)  # Thickness of the rectangle
    return img

def process_images(input_folder, output_folder):
    """
    Process all images in the input folder, detect objects, and save labeled images 
    to the output folder.
    
    Args:
        input_folder: Path to the folder containing input images.
        output_folder: Path to the folder where labeled images will be saved.
    """
    model = get_model()  # Load the YOLO model
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpeg', '.jpg', '.png')):  # Check for valid image extensions
            img_path = os.path.join(input_folder, filename)  # Full path to the input image
            
            # Read the image
            img = cv2.imread(img_path)
            
            # Run the model to get the detection results
            results = model(img_path)
            
            # Plot the bounding boxes on the image
            img_with_bboxes = plot_bboxes(results, img)
            
            # Save the labeled image to the output folder
            output_path = os.path.join(output_folder, f"labeled_{filename}")
            cv2.imwrite(output_path, img_with_bboxes)  # Save the image
            print(f"Image saved: {output_path}")  # Print the save path

# Define the input and output folders
input_folder = 'AMR'  # Folder containing input images
output_folder = 'AMR Labeled_Images'  # Folder to save labeled images

# Process all images in the input folder
process_images(input_folder, output_folder)
