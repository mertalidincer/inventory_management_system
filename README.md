# Inventory Management System

This project implements an inventory management system that detects shelves and counts the number of products on each shelf using the YOLO (You Only Look Once) object detection model. The system processes images to identify products and their respective shelves, providing a visual representation and product counts.

## Features

- Load and optimize a YOLO model for object detection.
- Automatically detect shelves and count products based on bounding box coordinates.
- Visualize detected shelves and product counts on the original image.
- Handle cases where no products are detected gracefully.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLO

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/inventory_management_system.git
   cd inventory_management_system
   ```

2. Install the required packages:
   ```
   pip install opencv-python numpy ultralytics
   ```

3. Download the YOLO model weights and place them in the `models` directory as `best.pt`.

## Usage

1. Place your input image in the project directory.
2. Modify the `total_number_of_products_on_each_shelf.py` file to specify the image filename in the `model` function call.
3. Run the script:
   ```
   python total_number_of_products_on_each_shelf.py
   ```

4. The processed image will be displayed with detected shelves and product counts.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

