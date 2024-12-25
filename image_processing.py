import numpy as np
import math
import cv2

def convert_to_grayscale(image):
    np_image = np.array(image)
    grayscale_image = np.dot(np_image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale_image.astype(np.uint8)

def canny_edge_detection(image, low_threshold, high_threshold):
    gray_image = convert_to_grayscale(image)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    
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

def dilate(image, kernel_size=3):
    np_image = np.array(image)
    dilated_image = np.zeros_like(np_image)
    
    pad = kernel_size // 2
    padded_image = np.pad(np_image, pad, mode='constant', constant_values=0)
    
    for i in range(pad, padded_image.shape[0] - pad):
        for j in range(pad, padded_image.shape[1] - pad):
            dilated_image[i - pad, j - pad] = np.max(padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1])
    
    return dilated_image

def hough_lines_p(image, rho, theta, threshold, min_line_length, max_line_gap):
    np_image = np.array(image)
    width, height = np_image.shape
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta))
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.int)
    y_idxs, x_idxs = np.nonzero(np_image)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for t_idx in range(len(thetas)):
            rho = int(round(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx]))) + diag_len
            accumulator[rho, t_idx] += 1
    
    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator > threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Calculate the length of the line segment
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if line_length >= min_line_length:
            lines.append((x1, y1, x2, y2))
    
    # Merge lines that are close to each other
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        merged = False
        for merged_line in merged_lines:
            mx1, my1, mx2, my2 = merged_line
            if (np.sqrt((x1 - mx1) ** 2 + (y1 - my1) ** 2) < max_line_gap and
                np.sqrt((x2 - mx2) ** 2 + (y2 - my2) ** 2) < max_line_gap):
                merged_line = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                merged = True
                break
        if not merged:
            merged_lines.append(line)
    
    return merged_lines

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