from ultralytics import YOLO 
import numpy as np 
import cv2

def get_model():
    # Load the YOLO model and fuse layers for optimization
    model = YOLO('models/best.pt')
    model.fuse()
    return model

# Determine if two boxes are on the same shelf
def boxes_are_on_same_shelf(box1, box2, y_threshold, x_threshold):
    # Calculate horizontal distance
    x_distance = min(abs(box1[2] - box2[0]), abs(box2[2] - box1[0]))
    # Calculate vertical center difference
    y_distance = abs((box1[1] + box1[3]) // 2 - (box2[1] + box2[3]) // 2)
    # Check if the boxes are close enough to be on the same shelf
    return x_distance < x_threshold or y_distance < y_threshold

def auto_detect_shelves_and_count_dynamic(results):
    """
    Automatically detects shelves and counts the number of products on each shelf 
    based on the bounding box coordinates of detected objects.
    Parameters:
        results (list): A list of detection results, where `results[0]` contains 
                        the original image and bounding box information. 
                        Each bounding box is represented as [x_min, y_min, x_max, y_max].
    Returns:
        numpy.ndarray: The original image with shelves and products processed. 
                       If no products are detected, the original image is returned unchanged.
    Notes:
        - The function calculates average box dimensions to determine thresholds 
          for grouping boxes into shelves.
        - Boxes are sorted by their vertical center (y-axis) to facilitate shelf grouping.
        - The function uses a helper function `boxes_are_on_same_shelf` to decide 
          whether two boxes belong to the same shelf based on their proximity.
    Raises:
        ValueError: If the input `results` is not in the expected format.
    """
    # Copy the original image
    img = results[0].orig_img.copy()
    # Extract bounding boxes and convert to integer format
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32)

    # If no products are detected, return the original image
    if len(boxes) == 0:
        print("No products detected.")
        return img
    
    # Determine thresholds based on average box dimensions
    avg_box_height = np.mean([box[3] - box[1] for box in boxes])
    avg_box_width = np.mean([box[0] - box[2] for box in boxes])
    y_threshold = avg_box_height * 0.7
    x_threshold = avg_box_width * 0.9

    # Sort boxes by their vertical center (y-axis)
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)

    shelf_groups = []  # To store shelf groups
    current_group = [boxes[0]]  # Start with the first box in the current group

    # Group boxes into shelves
    for i in range(1, len(boxes)):
        prev_box = current_group[-1]  # Last box in the current group
        curr_box = boxes[i]  # Current box being processed

        # Check if the current box belongs to the same shelf as the previous box
        if boxes_are_on_same_shelf(prev_box, curr_box, y_threshold, x_threshold): 
            current_group.append(curr_box)  # Add to the current shelf group
        else:
            shelf_groups.append(current_group)  # Save the current group as a shelf
            current_group = [curr_box]  # Start a new shelf group

    # Add the last group if it exists
    if current_group:
        shelf_groups.append(current_group)

    shelf_counts = {}  # To store the count of products on each shelf
    label_map = {}  # To map boxes to their respective shelf numbers
    shelf_num = 1  # Initialize shelf number

    # Draw shelf lines and count products
    for group in shelf_groups:
        tops = [box[1] for box in group]
        bottoms = [box[3] for box in group]
        upper_line = int(np.mean(tops))  # Calculate the upper line of the shelf
        lower_line = int(np.mean(bottoms))  # Calculate the lower line of the shelf

        # Draw the upper and lower lines of the shelf
        cv2.line(img, (0, upper_line), (img.shape[1], upper_line), (0, 0, 255), 3)
        cv2.putText(img, f"Shelf {shelf_num}", (10, upper_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.line(img, (0, lower_line), (img.shape[1], lower_line), (255, 0, 255), 3)

        # Count the number of products on the shelf
        shelf_counts[f"Shelf {shelf_num}"] = len(group)
        for box in group:
            label_map[tuple(box)] = shelf_num  # Map the box to the shelf number
        shelf_num += 1

    # Draw bounding boxes and annotate with shelf numbers
    for box in boxes:
        x1, y1, x2, y2 = box
        shelf_id = label_map.get(tuple(box), 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        if shelf_id:
            cv2.putText(img, f"Shelf {shelf_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)

    # Print the total number of shelves and product counts
    print(f"Total number of shelves: {len(shelf_counts)}")
    for shelf, count in shelf_counts.items():
        print(f"{shelf}: {count} products")

    return img

if __name__ == '__main__':
    # Load the YOLO model
    model = get_model()
    # Perform detection on the input image
    results = model('2_62_8_52_20220830_090200_.jpg')
    # Process the results to detect shelves and count products
    img = auto_detect_shelves_and_count_dynamic(results)

    # Display the processed image
    cv2.imshow('Shelves and Product Count', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
