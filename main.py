import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import math

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
    blurred_image = gaussian_blur(gray_image)
    G, theta = sobel_filters(blurred_image)
    non_max_img = non_maximum_suppression(G, theta)
    threshold_img, weak, strong = threshold(non_max_img)
    img_final = hysteresis(threshold_img, weak, strong)
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

if __name__ == "__main__":
    image_path = 'test.jpg'
    image = Image.open(image_path)
    
    edges = canny_edge_detection(image)
    
    top_quads = find_largest_quadrilaterals(edges, top_n=10)
    
    if top_quads:
        gray_image = rgb_to_grayscale(image)
        gray_array = np.array(gray_image)
        variances = []
        
        for quad in top_quads:
            var = evaluate_quadrilateral(gray_array, quad)
            variances.append((var, quad))
        
        variances.sort(key=lambda x: x[0])
        best_quad = variances[0][1]
        
        result_image = image.convert("RGB")
        result_image = draw_quadrilateral(result_image, best_quad, color="red", width=3)
    else:
        print("Yeterli dörtgen bulunamadı.")
        result_image = image.convert("RGB")
    
    output_path = 'output.jpg'
    result_image.save(output_path)
    result_image.show()