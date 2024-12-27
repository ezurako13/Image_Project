import numpy as np
import math
import cv2


def rgb_to_hsi(image):
    np_image = np.array(image) / 255.0
    hsi_image = np.zeros_like(np_image)
    
    r, g, b = np_image[..., 0], np_image[..., 1], np_image[..., 2]
    intensity = (r + g + b) / 3.0
    
    min_rgb = np.min(np_image, axis=-1)
    saturation = 1 - (3 / (r + g + b + 1e-10)) * min_rgb
    saturation[intensity == 0] = 0
    
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-10))
    hue = np.where(b <= g, theta, 2 * np.pi - theta)
    hue /= 2 * np.pi
    
    hsi_image[..., 0] = hue
    hsi_image[..., 1] = saturation
    hsi_image[..., 2] = intensity
    
    return hsi_image

def convert_to_grayscale_from_hsi(image):
    hsi_image = rgb_to_hsi(image)
    intensity = hsi_image[..., 2]
    grayscale_image = (intensity * 255).astype(np.uint8)
    return grayscale_image

def convert_to_grayscale(image):
    np_image = np.array(image)
    # grayscale_image = np.dot(np_image[..., :3], [0.299, 0.587, 0.114])    # RGB to Grayscale
    grayscale_image = np.dot(np_image[..., :3], [0.114, 0.587, 0.299])  # BGR to Grayscale (like OpenCV)
    return grayscale_image.astype(np.uint8)

def gaussian_blur(image, kernel_size, sigma):
    def gaussian_kernel(size, sigma):
        """Generates a Gaussian kernel."""
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
            ), (size, size)
        )
        return kernel / np.sum(kernel)
    
    np_image = np.array(image)
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    padded_image = np.pad(np_image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    blurred_image = np.zeros_like(np_image)
    
    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            blurred_image[i - pad, j - pad] = np.sum(padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1] * kernel)
    
    return blurred_image.astype(np.uint8)

def canny_edge_detection(image, low_threshold, high_threshold):
    gray_image = convert_to_grayscale(image)
    blurred_image = gaussian_blur(gray_image, 5, 1.4)
    
    gx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    gradient_direction = np.arctan2(gy, gx)
    
    nms_image = np.zeros_like(gradient_magnitude)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            q = 255
            r = 255
            
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]
            
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                nms_image[i, j] = gradient_magnitude[i, j]
            else:
                nms_image[i, j] = 0
    
    high_threshold = nms_image.max() * high_threshold
    low_threshold = high_threshold * low_threshold
    
    res = np.zeros_like(nms_image)
    
    strong_i, strong_j = np.where(nms_image >= high_threshold)
    weak_i, weak_j = np.where((nms_image <= high_threshold) & (nms_image >= low_threshold))
    
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 75
    
    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            if (res[i, j] == 75):
                if ((res[i + 1, j - 1] == 255) or (res[i + 1, j] == 255) or (res[i + 1, j + 1] == 255)
                        or (res[i, j - 1] == 255) or (res[i, j + 1] == 255)
                        or (res[i - 1, j - 1] == 255) or (res[i - 1, j] == 255) or (res[i - 1, j + 1] == 255)):
                    res[i, j] = 255
                else:
                    res[i, j] = 0
    
    return res.astype(np.uint8)

def dilate(image, kernel, iterations=1):
    np_image = np.array(image)
    if np_image.ndim == 2:  # Grayscale image
        np_image = np_image[:, :, np.newaxis]
    
    kernel = np.array(kernel)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(np_image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    dilated_image = np.zeros_like(np_image)
    
    for _ in range(iterations):
        for i in range(pad, padded_image.shape[0] - pad):
            for j in range(pad, padded_image.shape[1] - pad):
                for k in range(np_image.shape[2]):  # Iterate over channels
                    region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1, k]
                    dilated_image[i - pad, j - pad, k] = np.max(region * kernel)
        padded_image = np.pad(dilated_image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    if dilated_image.shape[2] == 1:  # If the image was originally grayscale
        dilated_image = dilated_image[:, :, 0]
    
    return dilated_image

# import numpy as np
from numba import jit
# import random

@jit(nopython=True)
def _hough_line_prob(img, rho, theta, threshold, line_length, line_gap):
    height, width = img.shape
    num_angle = int(np.pi / theta)
    num_rho = int(((width + height) * 2 + 1) / rho)
    
    # Compute trigonometric table
    cos_table = np.cos(np.arange(num_angle) * theta) / rho
    sin_table = np.sin(np.arange(num_angle) * theta) / rho
    
    # Initialize accumulator
    accum = np.zeros((num_angle, num_rho), dtype=np.int32)
    
    # Collect non-zero points
    points = np.column_stack(np.nonzero(img))
    mask = np.copy(img).astype(bool)
    
    lines = []
    shift = 16
    
    # Process points in random order
    np.random.shuffle(points)
    
    for point in points:
        y, x = point
        
        if not mask[y, x]:
            continue
            
        # Update accumulator
        max_val = 0
        max_n = 0
        
        for n in range(num_angle):
            r = int(round(x * cos_table[n] + y * sin_table[n]))
            r += (num_rho - 1) // 2
            accum[n, r] += 1
            if accum[n, r] > max_val:
                max_val = accum[n, r]
                max_n = n
                
        if max_val < threshold:
            continue
            
        # Find line segment
        a = -sin_table[max_n]
        b = cos_table[max_n]
        x0, y0 = x, y
        
        if abs(a) > abs(b):
            dx = 1 if a > 0 else -1
            dy = int(round(b * (1 << shift) / abs(a)))
            y0 = (y0 << shift) + (1 << (shift-1))
            xflag = True
        else:
            dy = 1 if b > 0 else -1
            dx = int(round(a * (1 << shift) / abs(b)))
            x0 = (x0 << shift) + (1 << (shift-1))
            xflag = False
            
        line_ends = []
        
        # Walk in both directions
        for k in range(2):
            gap = 0
            x, y = x0, y0
            dx_k = dx if k == 0 else -dx
            dy_k = dy if k == 0 else -dy
            
            while True:
                if xflag:
                    j = x
                    i = y >> shift
                else:
                    j = x >> shift
                    i = y
                    
                if not (0 <= j < width and 0 <= i < height):
                    break
                    
                if mask[i, j]:
                    gap = 0
                    line_ends.append((j, i))
                elif gap > line_gap:
                    break
                else:
                    gap += 1
                    
                x += dx_k
                y += dy_k
                
        if len(line_ends) >= 2:
            x1, y1 = line_ends[0]
            x2, y2 = line_ends[-1]
            
            if abs(x2 - x1) >= line_length or abs(y2 - y1) >= line_length:
                lines.append((x1, y1, x2, y2))
                
                # Clear used points
                for x, y in line_ends:
                    mask[y, x] = False
                    
    return np.array(lines)

from numpy.random import default_rng
from tqdm import tqdm 

def hough_lines_prob(img, rho, theta, threshold, minLineLength, maxLineGap, max_lines=None):
    """
    Probabilistic Hough Transform implementation optimized for Python/NumPy
    
    Parameters:
    img: numpy.ndarray, binary input image
    rho: float, distance resolution of the accumulator in pixels
    theta: float, angle resolution of the accumulator in radians
    threshold: int, accumulator threshold. Only lines above this are returned
    line_length: int, minimum line length
    line_gap: int, maximum gap between line segments
    max_lines: int, maximum number of lines to return (optional)
    
    Returns:
    lines: numpy.ndarray, detected lines as (x1, y1, x2, y2)
    """
    height, width = img.shape
    
    # Initialize accumulator
    num_angles = int(np.round(np.pi / theta))
    num_rhos = int(np.round(((width + height) * 2 + 1) / rho))
    accum = np.zeros((num_angles, num_rhos), dtype=np.int32)
    
    # Compute cosines and sines for all angles
    angles = np.arange(num_angles) * theta
    cos_table = np.cos(angles) / rho
    sin_table = np.sin(angles) / rho
    
    # Find non-zero points
    y_idxs, x_idxs = np.nonzero(img)
    points = np.column_stack((x_idxs, y_idxs))
    num_points = len(points)
    
    if num_points == 0:
        return np.array([], dtype=np.int32).reshape(0, 4)
    
    # Initialize mask for point tracking
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[y_idxs, x_idxs] = 1
    
    # Initialize RNG for random point selection
    rng = default_rng()
    
    # Store detected lines
    lines = []
    remaining_points = points.copy()

    # Initialize tqdm with the total number of iterations
    total_iterations = len(remaining_points) if max_lines is None else min(len(remaining_points), max_lines)
    progress_bar = tqdm(total=total_iterations)

    while len(remaining_points) > 0 and (max_lines is None or len(lines) < max_lines):
        # Choose random point
        idx = rng.integers(len(remaining_points))
        pt = remaining_points[idx]
        
        # Update accumulator for this point
        rho_values = np.round(pt[0] * cos_table + pt[1] * sin_table).astype(np.int32)
        rho_values += (num_rhos - 1) // 2
        
        # Update accumulator and find maximum
        for angle_idx, r in enumerate(rho_values):
            accum[angle_idx, r] += 1
        
        max_val = accum.max()
        if max_val < threshold:
            remaining_points = np.delete(remaining_points, idx, axis=0)
            continue
        
        # Find the angle with maximum votes
        angle_idx, rho_idx = np.unravel_index(accum.argmax(), accum.shape)
        
        # Calculate line parameters
        theta_max = angle_idx * theta
        cos_t = cos_table[angle_idx] * rho
        sin_t = sin_table[angle_idx] * rho
        
        # Walk along the line in both directions
        x0, y0 = pt
        
        # Determine line direction
        if abs(cos_t) > abs(sin_t):
            dx = np.sign(cos_t)
            dy = sin_t / abs(cos_t)
            x_flag = True
        else:
            dy = np.sign(sin_t)
            dx = cos_t / abs(sin_t)
            x_flag = False
            
        # Find line endpoints
        line_ends = []
        for direction in [1, -1]:
            x, y = x0, y0
            gap = 0
            
            while True:
                ix = int(round(x))
                iy = int(round(y))
                
                if (ix < 0 or ix >= width or iy < 0 or iy >= height):
                    break
                
                if mask[iy, ix]:
                    gap = 0
                    line_ends.append((ix, iy))
                    mask[iy, ix] = 0
                else:
                    gap += 1
                    if gap > maxLineGap:
                        break
                
                x += dx * direction
                y += dy * direction
        
        if len(line_ends) >= 2:
            x1, y1 = line_ends[0]
            x2, y2 = line_ends[-1]
            
            # Check line length
            if (abs(x2 - x1) >= minLineLength or abs(y2 - y1) >= minLineLength):
                lines.append([x1, y1, x2, y2])
                
                # Update accumulator (remove used points)
                for x, y in line_ends:
                    rho_values = np.round(x * cos_table + y * sin_table).astype(np.int32)
                    rho_values += (num_rhos - 1) // 2
                    for angle_idx, r in enumerate(rho_values):
                        accum[angle_idx, r] -= 1
        
        remaining_points = np.delete(remaining_points, idx, axis=0)

        progress_bar.update(1)
    
    progress_bar.close()

    return np.array(lines, dtype=np.int32)

def HoughLinesP(img, rho=1, theta=np.pi/180, threshold=50, 
                    minLineLength=50, maxLineGap=10):
    """
    Probabilistic Hough Transform
    Parameters:
        img: Binary input image
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Accumulator threshold parameter
        line_length: Minimum line length
        line_gap: Maximum gap between line segments
    Returns:
        lines: Array of detected line segments
    """
    return _hough_line_prob(img, rho, theta, threshold, minLineLength, maxLineGap)

def flood_fill(image, seed_point, new_value):
    np_image = np.array(image)
    original_value = np_image[seed_point]
    stack = [seed_point]
    
    while stack:
        x, y = stack.pop()
        if np_image[x, y] == original_value:
            np_image[x, y] = new_value
            if x > 0:
                stack.append((x - 1, y))
            if x < np_image.shape[0] - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < np_image.shape[1] - 1:
                stack.append((x, y + 1))
    
    return np_image

def active_contour(image, snake, alpha=0.1, beta=0.1, gamma=0.1, iterations=100):
    np_image = np.array(image)
    for _ in range(iterations):
        for i in range(len(snake)):
            x, y = snake[i]
            fx = np.gradient(np_image[:, y])
            fy = np.gradient(np_image[x, :])
            snake[i] = (x + gamma * fx[x], y + gamma * fy[y])
    return snake

def rgb_to_hsv(image):
    np_image = np.array(image) / 255.0
    hsv_image = np.zeros_like(np_image)
    
    r, g, b = np_image[..., 0], np_image[..., 1], np_image[..., 2]
    maxc = np.max(np_image, axis=-1)
    minc = np.min(np_image, axis=-1)
    v = maxc
    s = (maxc - minc) / maxc
    s[maxc == 0] = 0
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    
    h = np.zeros_like(maxc)
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[b == maxc] = 4.0 + gc[b == maxc] - rc[b == maxc]
    h[minc == maxc] = 0.0
    h = (h / 6.0) % 1.0
    
    hsv_image[..., 0] = h
    hsv_image[..., 1] = s
    hsv_image[..., 2] = v
    
    return hsv_image

def color_mask(image, lower_hsv, upper_hsv):
    hsv_image = rgb_to_hsv(image)
    mask = np.all((hsv_image >= lower_hsv) & (hsv_image <= upper_hsv), axis=-1)
    masked_image = np.zeros_like(hsv_image)
    masked_image[mask] = hsv_image[mask]
    return (masked_image * 255).astype(np.uint8)