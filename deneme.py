import image_processing as xd
import numpy as np
import cv2

image = cv2.imread('./inputs/input.jpg')


edges_cv = cv2.Canny(image, 50, 200, apertureSize=3)
cv2.imwrite("outputs/Edges.png", edges_cv)

edges_xd = xd.canny_edge_detection(image, 50, 200)
cv2.imwrite("outputs/Edges_xd.png", edges_xd)

