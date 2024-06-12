# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:46:44 2024

@author: ROG ZEPHYRUS
"""

from ultralytics import YOLO

def main():
    # Load the pre-trained model
    model = YOLO("C:/Users/ROG ZEPHYRUS/MiniProject-Traffic Detection/runs/detect/train29/weights/best.pt") 
    # Make predictions on new data
    results = model("C:/Users/ROG ZEPHYRUS/Downloads/15_1.jpg")  # Provide the path to your image
    # Process the results as needed
    print(results)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk
if __name__ == '__main__':
    main()
