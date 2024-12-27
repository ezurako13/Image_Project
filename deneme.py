import image_processing as xd
import numpy as np
import cv2


def test_grayscale():
	image = cv2.imread('./inputs/input.jpg')
	grayscale_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("outputs/Grayscale.png", grayscale_cv)

	# grayscale_xd = xd.convert_to_grayscale_from_hsi(image)
	grayscale_xd = xd.convert_to_grayscale(image)
	cv2.imwrite("outputs/Grayscale_xd.png", grayscale_xd)

def test_gaussian_blur():
	image = cv2.imread('./outputs/Grayscale.png')
	blurred_cv = cv2.GaussianBlur(image, (5, 5), 1.4)
	cv2.imwrite("outputs/Blurred.png", blurred_cv)

	blurred_xd = xd.gaussian_blur(image, 5, 1.4)
	cv2.imwrite("outputs/Blurred_xd.png", blurred_xd)

def test_canny():
	image = cv2.imread('./outputs/Blurred Image.png.jpg')

	edges_cv = cv2.Canny(image, 50, 200, apertureSize=3)
	cv2.imwrite("outputs/Edges.png", edges_cv)

	edges_xd = xd.canny_edge_detection(image, 50, 200)
	cv2.imwrite("outputs/Edges_xd.png", edges_xd)

def test_dilate():
	image = cv2.imread('./outputs/Edges.png')
	kernel = np.ones((5, 5), np.uint8)
	dilated_cv = cv2.dilate(image, kernel, iterations=1)
	cv2.imwrite("outputs/Dilated.png", dilated_cv)

	dilated_xd = xd.dilate(image, kernel, 1)
	cv2.imwrite("outputs/Dilated_xd.png", dilated_xd)

def test_hough_transform():
	image = cv2.imread('./outputs/Edges.png', cv2.IMREAD_GRAYSCALE)
	_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	
	lines_cv = cv2.HoughLinesP(image, 1, np.pi/180, threshold=300, minLineLength=50, maxLineGap=10)
	hough_lines = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
	for line in lines_cv:
		x1, y1, x2, y2 = line[0]
		cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.imwrite("outputs/Hough_Lines.png", hough_lines)

	lines_xd = xd.hough_lines_prob(image, 1, np.pi/180, threshold=300, minLineLength=50, maxLineGap=10)
	hough_lines = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
	for line in lines_xd:
		x1, y1, x2, y2 = line
		cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.imwrite("outputs/Hough_Lines_xd.png", hough_lines)

test_hough_transform()