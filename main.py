import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import math
import cv2
import image_processing as xd


def rgb_to_grayscale(image):
    return ImageOps.grayscale(image)

def gaussian_blur(image, sigma=1.4):
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def sobel_filters(image):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    Ix = image.filter(ImageFilter.Kernel((3, 3), Kx.flatten(), scale=1))
    Iy = image.filter(ImageFilter.Kernel((3, 3), Ky.flatten(), scale=1))
    
    Ix = np.array(Ix, dtype=float)
    Iy = np.array(Iy, dtype=float)
    
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    theta = np.arctan2(Iy, Ix)
    
    return G, theta

def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j + 1]
                    r = G[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = G[i + 1, j - 1]
                    r = G[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = G[i + 1, j]
                    r = G[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = G[i - 1, j - 1]
                    r = G[i + 1, j + 1]
                
                if (G[i, j] >= q) and (G[i, j] >= r):
                    Z[i, j] = G[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z

def threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    highThreshold = image.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(image >= highThreshold)
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res, weak, strong

def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] == weak:
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                    or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                    or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image):
    gray_image = rgb_to_grayscale(image)
    gray_image.save("outputs/Gray Image.png")
    blurred_image = gaussian_blur(gray_image)
    blurred_image.save("outputs/Blurred Image.png")
    G, theta = sobel_filters(blurred_image)
    non_max_img = non_maximum_suppression(G, theta)
    #show non-maximum suppressed image
    Image.fromarray(np.uint8(non_max_img)).save("outputs/Non-Maximum Suppressed Image.png")
    threshold_img, weak, strong = threshold(non_max_img)
    #show threshold image
    Image.fromarray(np.uint8(threshold_img)).save("outputs/Threshold Image.png")
    img_final = hysteresis(threshold_img, weak, strong)
    #show final image
    Image.fromarray(np.uint8(img_final)).save("outputs/Final Image.png")
    return img_final

def compute_convex_hull(points):
    def polar_angle(p0, p1=None):
        if p1 is None:
            p1 = anchor
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return math.atan2(y_span, x_span)
    
    def distance(p0, p1=None):
        if p1 is None:
            p1 = anchor
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return y_span ** 2 + x_span ** 2
    
    sorted_points = sorted(points, key=lambda x: (x[1], x[0]))
    anchor = sorted_points[0]
    
    sorted_points = sorted(sorted_points[1:], key=lambda point: (polar_angle(point), -distance(point)))
    
    hull = [anchor, sorted_points[0]]
    for s in sorted_points[1:]:
        while len(hull) >= 2 and cross_product(np.subtract(hull[-1], hull[-2]), np.subtract(s, hull[-2])) <= 0:
            hull.pop()
        hull.append(s)
    return np.array(hull)

def cross_product(a, b):
    return a[0]*b[1] - a[1]*b[0]

def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def find_largest_quadrilaterals(edges, top_n=10):
    edge_coords = np.argwhere(edges == 255)
    if len(edge_coords) < 4:
        return []
    
    hull_points = compute_convex_hull(edge_coords.tolist())
    quadrilaterals = []
    n = len(hull_points)
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    quad = [hull_points[i], hull_points[j], hull_points[k], hull_points[l]]
                    area = polygon_area(np.array(quad))
                    quadrilaterals.append((area, quad))
                    
    quadrilaterals.sort(reverse=True, key=lambda x: x[0])
    top_quadrilaterals = []
    seen_corners = set()
    
    for area, quad in quadrilaterals:
        corners = tuple(map(tuple, sorted(map(tuple, quad))))
        if corners not in seen_corners:
            seen_corners.add(corners)
            top_quadrilaterals.append(quad)
            if len(top_quadrilaterals) == top_n:
                break
    
    return top_quadrilaterals

def get_line_pixels(img_array, start, end, width=3):
    pixels = []
    x1, y1 = start
    x2, y2 = end
    
    length = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    if length == 0:
        return []
    
    for i in range(length):
        t = i / length
        x = int(x1 * (1 - t) + x2 * t)
        y = int(y1 * (1 - t) + y2 * t)
        
        for dx in range(-width//2, width//2 + 1):
            for dy in range(-width//2, width//2 + 1):
                if (0 <= x + dx < img_array.shape[0] and 
                    0 <= y + dy < img_array.shape[1]):
                    pixels.append(img_array[x + dx, y + dy])
    return np.array(pixels)

def evaluate_quadrilateral(img_array, quad):
    corner_values = []
    edge_values = []
    
    for i in range(4):
        start = quad[i]
        end = quad[(i + 1) % 4]
        pixels = get_line_pixels(img_array, start, end, width=3)
        if len(pixels) == 0:
            return float('inf')
        edge_values.append(pixels)
        corner_values.append(img_array[start[0], start[1]])
    
    if len(set(corner_values)) != 1:
        return float('inf')
    
    for edge in edge_values:
        mean_value = np.mean(edge)
        if np.any(np.abs(edge - mean_value) > mean_value * 0.1):
            return float('inf')
    
    edge_means = [np.mean(edge) for edge in edge_values]
    if np.any(np.abs(np.diff(edge_means)) > np.mean(edge_means) * 0.1):
        return float('inf')
    
    return np.var(corner_values)

def draw_quadrilateral(image, quad, color="red", width=3):
    draw = ImageDraw.Draw(image)
    quad_points = [tuple(point[::-1]) for point in quad]
    for i in range(4):
        start = quad_points[i]
        end = quad_points[(i + 1) % 4]
        draw.line([start, end], fill=color, width=width)
    return image

def find_colored_region_corners(image_path, lower_color, upper_color):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Hata: Görüntü yüklenemedi.")
        return []
    
    # Görüntüyü HSV formatına çevir
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Renk aralığına göre maske oluştur
    mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))
    
    # Maskede konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for cnt in contours:
        # Minimum alanı kaplayan dikdörtgeni bul
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Köşe noktalarını tamsayıya çevir
        
        corners.append(box)
    
    return corners

def find_blue_contours(image_path, output_image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the blue color range in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for the blue color
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Show the contours 
    Image.fromarray(mask_blue).save("outputs/mask_blueddd.png")
    
    # Draw contours on the original image with white color
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

    # Create a mask for white pixels
    white_mask = np.all(image == [255, 255, 255], axis=-1)
    white_pixels = np.argwhere(white_mask)

    # Save the result
    cv2.imwrite(output_image_path, image)

    return contours, white_pixels

def flood_fill(image, seed_point, new_color, scale=10):
    # Convert PIL image to numpy array
    image_np = np.array(image)

    if len(image_np.shape) == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_np

    h, w = gray_image.shape[:2]

    # Create a mask for flood fill
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # print("seed_point ", seed_point)

    # print("seed_point ", seed_point)
    # print("scale ", scale)
    # find nearest black pixel in all directions
    for y in range(seed_point[0] - scale*5, seed_point[0] + scale*5):
        # print("y ", y, "x ", seed_point[1])
        for x in range(seed_point[1] - scale*5, seed_point[1] + scale*5):
            # print("x ", x)
            # print(y, x)
            # if gray_image[y, x] == 0:
                # seed_point = (y, x)
                # print("seed_point ", seed_point)
                
                # seed_point = (seed_point[1], seed_point[0])

            # Perform flood fill
            cv2.floodFill(image_np, mask, (x, y), new_color, (10,), (10,), cv2.FLOODFILL_FIXED_RANGE)

            # image_np[y, x] = new_color
                # break
        # if gray_image[y, x] == 0:
        #     break
    
    # print("seed_point ", seed_point)


    # Convert numpy array back to PIL image
    return image_np

def restore_blue_pixels(original_image_path, flood_filled_image_path, output_image_path):
    # Load the images
    flood_filled_image = cv2.imread(flood_filled_image_path)
    original_image = cv2.imread(original_image_path)

    # Convert the flood filled image to HSV
    hsv_flood_filled = cv2.cvtColor(flood_filled_image, cv2.COLOR_BGR2HSV)

    # Define the blue color range in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for the blue color
    mask_blue = cv2.inRange(hsv_flood_filled, lower_blue, upper_blue)

    # Extract the blue regions from the original image
    blue_regions = cv2.bitwise_and(original_image, original_image, mask=mask_blue)

    # Create an inverse mask to remove the blue regions from the flood filled image
    mask_blue_inv = cv2.bitwise_not(mask_blue)
    flood_filled_no_blue = cv2.bitwise_and(flood_filled_image, flood_filled_image, mask=mask_blue_inv)

    # Combine the images to place the blue regions back into the original image
    result_image = cv2.add(flood_filled_no_blue, blue_regions)

    # Save the result
    cv2.imwrite(output_image_path, result_image)

def draw_boundaries_on_original(original_image_path, coordinates, output_image_path):
    # Load the original image
    original_image = cv2.imread(original_image_path)

    if original_image is None:
        raise FileNotFoundError(f"Cannot open {original_image_path}")

    # Iterate over the coordinates and set the pixels to white
    for coord in coordinates:
        y, x = coord  # Unpack only y and x
        original_image[y, x] = [255, 255, 255]

    # Save the result
    cv2.imwrite(output_image_path, original_image)

def create_black_image_with_white_pixels(image_name, width, height, coordinates):
    # Create a black image
    image = np.zeros((height, width, 3), np.uint8)

    # Iterate over the coordinates and set the pixels to white
    for coord in coordinates:
        y, x = coord  # Unpack only y and x
        image[y, x] = [255, 255, 255]

    # Save the result
    cv2.imwrite(image_name, image)

def gurpinar(image_path):
    # image_path = 'input2.jpg'
    image = Image.open(image_path)
    
    # edges = canny_edge_detection(image)

    image_height = image.size[1]
    image_width = image.size[0]

    scale = min(image_height, image_width) // 120

    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Show blurred image
    Image.fromarray(gray).save("outputs/Blurred Image.png")
    edges_cv = cv2.Canny(gray, 50, 200, apertureSize=3)

    # Show edges
    Image.fromarray(edges_cv).save("outputs/Edges.png")

    # Create a kernel with plus shape
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    # Perform morphological operations
    edges_cv = cv2.dilate(edges_cv, kernel, iterations=2)


    # Show morphed edges
    Image.fromarray(edges_cv).save("outputs/Edges.png")

    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(edges_cv, 1, np.pi / 720, threshold=scale*30, minLineLength=scale*5, maxLineGap=scale)
 
    # Show the hough lines
    hough_lines = np.zeros((image_cv.shape[0], image_cv.shape[1], 3), np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    hough_lines = Image.fromarray(hough_lines)
    hough_lines.save("outputs/Lines.png")


    # # Draw lines on the image
    # image_cv = np.array(image_cv)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # image_cv = Image.fromarray(image_cv)
    # image_cv.save("outputs/Hough Lines.png")


    hough_lines = cv2.dilate(np.array(hough_lines), kernel, iterations=3)
    image_cv = Image.fromarray(hough_lines)

    # image_cv = hough_lines

    # # Create RGB image from binary image
    # image_cv = cv2.cvtColor(edges_cv, cv2.COLOR_GRAY2BGR)
    # image_cv = Image.fromarray(image_cv)

    width, height = image_cv.size
    # seed_point = (width // 2, height // 2)  # Use the center of the image as the seed point
    seed_point = (height // 2, width // 2)  # Use the center of the image as the seed point
    image_cv = flood_fill(image_cv, seed_point, (0, 0, 255), scale=scale)
    
    
    # # Apply blue mask ON THE IMAGE
    # lower = np.array([0, 255, 0])
    # upper = np.array([0, 255, 0])
    # mask = cv2.inRange(image_cv, lower, upper)
    # masked = cv2.bitwise_and(image_cv, image_cv, mask=mask)
    # result = image_cv - masked
    # image_cv = cv2.dilate(np.array(result), np.ones((3, 3), np.uint8), iterations=1)


    Image.fromarray(image_cv).save("outputs/flood_fill.png")
    restore_blue_pixels(image_path, "outputs/flood_fill.png", "outputs/res.png")
    outputBoundryPath = 'outputs/outputBoundry.png'
    contoursFinded,white_pixels = find_blue_contours("outputs/flood_fill.png", outputBoundryPath)
    print(f"Number of contours found: {len(contoursFinded)} num of white pixels: {len(white_pixels)}")
    finalOutput = 'outputs/finalOutput.png'
    
    image_path2 = image_path
    orginImage = Image.open(image_path2)
    draw_boundaries_on_original(image_path2, white_pixels, finalOutput)
    create_black_image_with_white_pixels("outputs/black_image.png", width, height, white_pixels)
    
    
    # image = Image.open("./mask_blueddd.png")
    # image_cv = np.array(image)
    # # # Apply active contour to Lines
    # # image_float = img_as_float(np.array(image))
    # # s = np.linspace(0, 2*np.pi, 400)
    # # init = np.array([image_float.shape[1]/2 + 100*np.cos(s), image_float.shape[0]/2 + 100*np.sin(s)]).T
    # # snake = active_contour(gaussian(image_float, 3), init, alpha=0.015, beta=10, gamma=0.001)

    # # # Draw the active contour on the image
    # # for i in range(len(snake) - 1):
    # #     cv2.line(image_cv, (int(snake[i, 0]), int(snake[i, 1])), (int(snake[i+1, 0]), int(snake[i+1, 1])), (255, 0, 0), 2)
    # # cv2.line(image_cv, (int(snake[-1, 0]), int(snake[-1, 1])), (int(snake[0, 0]), int(snake[0, 1])), (255, 0, 0), 2)

    # Image.fromarray(image_cv).show()
    # countours, _ = cv2.findContours(image_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # blank_image = np.zeros((image_cv.shape[0], image_cv.shape[1], 3), np.uint8)
    # image_cv = cv2.drawContours(blank_image, countours, -1, (0, 255, 0), 3)


    # flood_fill_cv = flood_fill(image_cv, seed_point, (0, 0, 255), scale=scale)
    # Image.fromarray(flood_fill_cv).show()
    # countours, _ = cv2.findContours(flood_fill_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image_cv = cv2.drawContours(blank_image, countours, -1, (255, 0, 0), 3)

    # image_cv = Image.fromarray(image_cv)
    # image_cv.save("outputs/Active Contour.png")

def run_tum_imagelar():
    image_paths = [ 
                    "input.jpg",
                    "input2.jpg",
                    "input3.jpg",
                    "input4.jpg",
                    "input5.jpg",
                    "input6.jpg",
                    "input7.jpg",
                    "input8.jpg",
                    "input9.jpg",
                    "input10.jpg",
                    "input11.jpg",
                    "input12.jpg",
                    "input14.jpg",
                    "input15.jpg"
                    ]
    
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        gurpinar("inputs/" + image_path)
        # print(f"Processing {image_path} is done.\n")
        # Wait for user input to continue
        input("Press Enter to continue...")


if __name__ == "__main__":
    run_tum_imagelar()
    # gurpinar("input7.jpg")

