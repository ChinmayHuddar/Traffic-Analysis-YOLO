# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:15:17 2024

@author: ROG ZEPHYRUS
"""


import cv2
import numpy as np
from PIL import Image

def divide_into_grids(image, rows, cols):
    height, width = image.shape[:2]
    grid_image = image.copy()
    
    # Compute grid size
    cell_width = width // cols
    cell_height = height // rows
    
    # Draw vertical grid lines
    for i in range(1, cols):
        x = i * cell_width
        cv2.line(grid_image, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Draw horizontal grid lines
    for j in range(1, rows):
        y = j * cell_height
        cv2.line(grid_image, (0, y), (width, y), (0, 255, 0), 2)
    
    return grid_image

# Load an image
image = cv2.imread('image.jpeg')


# Divide lanes image into grids
grid_image = divide_into_grids(image, 10, 20)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Grids', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()