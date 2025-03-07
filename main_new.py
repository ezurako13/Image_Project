import numpy as np
from PIL import Image
import cv2
import image_processing_new as xd
import re
import os

indexing=0
PAH=""
Cannyflag=False

def find_blue_contours(image_path, output_image_path):
    image = cv2.imread(image_path)

    hsv_image = xd.cvtColor(image, "COLOR_BGR2HSV")

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 1
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    Image.fromarray(mask_blue).save(PAH+"mask_blueddd.png")
    
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

    white_mask = np.all(image == [255, 255, 255], axis=-1)
    white_pixels = np.argwhere(white_mask)

    cv2.imwrite(output_image_path, image)

    return contours, white_pixels

def flood_fill(image, seed_point, new_color, scale=10):
    image_np = np.array(image)

    if len(image_np.shape) == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_np

    h, w = gray_image.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)


    for y in range(seed_point[0] - scale*5, seed_point[0] + scale*5):
        for x in range(seed_point[1] - scale*5, seed_point[1] + scale*5):
                

            xd.floodFill(image_np, mask, (x, y), new_color, (10,), (10,), cv2.FLOODFILL_FIXED_RANGE)

    


    return image_np

def restore_blue_pixels(original_image_path, flood_filled_image_path, output_image_path):
    flood_filled_image = cv2.imread(flood_filled_image_path)
    original_image = cv2.imread(original_image_path)

    hsv_flood_filled = cv2.cvtColor(flood_filled_image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(hsv_flood_filled, lower_blue, upper_blue)

    blue_regions = xd.bitwise_and(original_image, original_image, mask=mask_blue)

    mask_blue_inv = xd.bitwise_not(mask_blue)
    flood_filled_no_blue = xd.bitwise_and(flood_filled_image, flood_filled_image, mask=mask_blue_inv)

    result_image = xd.add(flood_filled_no_blue, blue_regions)

    cv2.imwrite(output_image_path, result_image)

def draw_boundaries_on_original(original_image_path, coordinates, output_image_path):
    original_image = cv2.imread(original_image_path)

    if original_image is None:
        raise FileNotFoundError(f"Cannot open {original_image_path}")

    for coord in coordinates:
        y, x = coord 
        original_image[y, x] = [255, 255, 255]

    cv2.imwrite(output_image_path, original_image)

def create_black_image_with_white_pixels(image_name, width, height, coordinates):
    image = np.zeros((height, width, 3), np.uint8)

    for coord in coordinates:
        y, x = coord
        image[y, x] = [255, 255, 255]

    cv2.imwrite(image_name, image)

def extract_index_from_path(image_path):
    match = re.search(r'input(\d+)\.jpg', image_path)
    if match:
        return int(match.group(1))
    return None

def gurpinar(image_path):
    global indexing
    global PAH
    global Cannyflag
    indexing = extract_index_from_path(image_path)

    print(f"Processing image {indexing}...")

    image = Image.open(image_path)
    if Cannyflag == False:
        outputPathForEachImage = 'WithoutCV2Cann_outputs/'+str(indexing)+"/"
    else:
        outputPathForEachImage = 'WithCV2Cann_outputs/'+str(indexing)+"/"

    PAH = outputPathForEachImage
    image_height = image.size[1]
    image_width = image.size[0]

    scale = min(image_height, image_width) // 120

    image_cv = np.array(image)
    gray = xd.cvtColor(image_cv, "COLOR_BGR2GRAY")
    os.makedirs(PAH, exist_ok=True)
    Image.fromarray(gray).save(outputPathForEachImage+"Grayscale Image.png")
    if Cannyflag == False:
        edges_cv = xd.canny_edge_detection(gray, 0.05, 0.15, 1.4)
    else:
        edges_cv = cv2.Canny(gray, 50, 200, apertureSize=3)
    Image.fromarray(edges_cv).save(outputPathForEachImage+"Edges.png")

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    edges_cv = xd.dilate(edges_cv,kernel=kernel, iterations=2)
    


    Image.fromarray(edges_cv).save(outputPathForEachImage+"Thick Edges.png")

    lines = cv2.HoughLinesP(edges_cv, 1, np.pi / 720, threshold=scale*30, minLineLength=scale*5, maxLineGap=scale)
 
    hough_lines = np.zeros((image_cv.shape[0], image_cv.shape[1], 3), np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    hough_lines = Image.fromarray(hough_lines)
    hough_lines.save(outputPathForEachImage+"Lines.png")


    hough_lines = xd.dilate(np.array(hough_lines), kernel, iterations=3)
    image_cv = Image.fromarray(hough_lines)
    image_cv.save(outputPathForEachImage+"Thick Hough Lines.png")


    width, height = image_cv.size
    seed_point = (height // 2, width // 2)
    image_cv = flood_fill(image_cv, seed_point, (0, 0, 255), scale=scale)


    Image.fromarray(image_cv).save(outputPathForEachImage+"flood_fill.png")
    restore_blue_pixels(image_path, outputPathForEachImage+"flood_fill.png", outputPathForEachImage+"res.png")
    outputBoundryPath = outputPathForEachImage+'outputBoundry.png'
    contoursFinded,white_pixels = find_blue_contours(outputPathForEachImage+"flood_fill.png", outputBoundryPath)
    print(f"Number of contours found: {len(contoursFinded)} num of white pixels: {len(white_pixels)}")
    finalOutput = outputPathForEachImage+'finalOutput.png'
    
    image_path2 = image_path
    orginImage = Image.open(image_path2)
    draw_boundaries_on_original(image_path2, white_pixels, finalOutput)
    create_black_image_with_white_pixels(outputPathForEachImage+"black_image.png", width, height, white_pixels)

def run_tum_imagelar():
    cwd = os.getcwd()
    global Cannyflag
    Cannyflag=False
    input_dir = cwd+'/inputs/'
    for image_path in os.listdir(input_dir):
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            gurpinar(os.path.join(input_dir, image_path))
    
    Cannyflag=True
    for image_path in os.listdir(input_dir):
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            gurpinar(os.path.join(input_dir, image_path))
            

if __name__ == "__main__":
    # gurpinar("inputs/input15.jpg")
    run_tum_imagelar()

