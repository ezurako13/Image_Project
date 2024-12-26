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
def filter2D(src, kernel):
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    padded_src = np.pad(src, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    dst = np.zeros_like(src)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            region = padded_src[i:i + k_height, j:j + k_width]
            dst[i, j] = np.sum(region * kernel)

    return dst

def gaussian_blur(src, ksize, sigma):
    k = (ksize - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    g = g / g.sum()
    return filter2D(src, g)

def Canny(src, low_thresh, high_thresh, apertureSize, L2gradient=False):
    assert src.dtype == np.uint8

    # Apply Gaussian blur
    src = gaussian_blur(src, 5, 1.4)

    size = src.shape
    dst = np.zeros(size, dtype=np.uint8)

    if (apertureSize & 1) == 0 or (apertureSize != -1 and (apertureSize < 3 or apertureSize > 7)):
        raise ValueError("Aperture size should be odd between 3 and 7")

    if apertureSize == 7:
        low_thresh /= 16.0
        high_thresh /= 16.0

    if low_thresh > high_thresh:
        low_thresh, high_thresh = high_thresh, low_thresh

    if L2gradient:
        low_thresh = min(32767.0, low_thresh)
        high_thresh = min(32767.0, high_thresh)
        if low_thresh > 0:
            low_thresh *= low_thresh
        if high_thresh > 0:
            high_thresh *= high_thresh

    low = int(np.floor(low_thresh))
    high = int(np.floor(high_thresh))

    # Sobel kernels
    if apertureSize == 3:
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    elif apertureSize == 5:
        Kx = np.array([[-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6], [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]], dtype=np.float32)
        Ky = np.array([[1, 4, 6, 4, 1], [2, 8, 12, 8, 2], [0, 0, 0, 0, 0], [-2, -8, -12, -8, -2], [-1, -4, -6, -4, -1]], dtype=np.float32)
    else:
        raise ValueError("Only apertureSize=3 or 5 is supported in this implementation")

    # Gradient calculation
    Ix = filter2D(src, Kx)
    Iy = filter2D(src, Ky)

    if L2gradient:
        magnitude = np.sqrt(Ix**2 + Iy**2)
    else:
        magnitude = np.abs(Ix) + np.abs(Iy)

    angle = np.arctan2(Iy, Ix) * (180 / np.pi)
    angle[angle < 0] += 180

    # Non-maximum suppression
    nms = np.zeros_like(magnitude, dtype=np.uint8)
    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0

    # Double threshold
    strong = 255
    weak = 75
    res = np.zeros_like(src, dtype=np.uint8)
    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms <= high) & (nms >= low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Edge tracking by hysteresis
    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            if res[i, j] == weak:
                if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong)
                        or (res[i, j - 1] == strong) or (res[i, j + 1] == strong)
                        or (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return res

def bitwise_and(image1, image2, mask=None):
    if mask is not None:
        return np.where(mask[:, :, None], image1 & image2, 0)
    return image1 & image2

def bitwise_not(image):
    return np.bitwise_not(image)

def add(image1, image2):
    height, width, channels = image1.shape
    result = [[[0] * channels for _ in range(width)] for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                result[i][j][k] = min(image1[i][j][k] + image2[i][j][k], 255)
    
    return np.array(result, dtype=np.uint8)

def floodFill(image, mask, seed_point, new_color, lo_diff, up_diff, flags):
    height, width, channels = image.shape
    x, y = seed_point
    original_color = image[y, x].tolist()
    stack = [(x, y)]
    mask[y, x] = 1

    while stack:
        x, y = stack.pop()
        image[y, x] = new_color

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] == 0:
                pixel_color = image[ny, nx].tolist()
                if all(abs(pixel_color[c] - original_color[c]) <= lo_diff[0] for c in range(channels)):
                    mask[ny, nx] = 1
                    stack.append((nx, ny))
def findContours(mask, mode, method):
    contours = []
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    def get_neighbors(x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))
        return neighbors

    def trace_contour(start_x, start_y):
        contour = []
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            contour.append((x, y))
            for nx, ny in get_neighbors(x, y):
                if mask[ny, nx] > 0 and not visited[ny, nx]:
                    stack.append((nx, ny))
        return contour

    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0 and not visited[y, x]:
                contour = trace_contour(x, y)
                if len(contour) > 10:  # Filter out small contours
                    contours.append(np.array(contour))

    return contours, None


def bgr_to_hsv(image):
    image = image.astype('float32') / 255.0
    hsv_image = np.zeros_like(image)
    
    b, g, r = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=-1)
    minc = np.min(image, axis=-1)
    delta = maxc - minc

    # Avoid division by zero
    s = np.zeros_like(maxc)
    mask = maxc != 0
    s[mask] = delta[mask] / maxc[mask]
    
    # Initialize hue with zeros
    h = np.zeros_like(maxc)
    
    # Compute hue for each channel where delta is not zero
    mask_delta = delta != 0
    mask_r = (maxc == r) & mask_delta
    mask_g = (maxc == g) & mask_delta
    mask_b = (maxc == b) & mask_delta

    # Hue calculation
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (2.0 + (b[mask_g] - r[mask_g]) / delta[mask_g])
    h[mask_b] = (4.0 + (r[mask_b] - g[mask_b]) / delta[mask_b])

    # Normalize hue to [0, 180]
    h = (h / 6.0) * 180.0

    # Scale saturation and value to [0, 255]
    s = s * 255.0
    v = maxc * 255.0

    hsv_image[..., 0] = h
    hsv_image[..., 1] = s
    hsv_image[..., 2] = v

    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
    
    return hsv_image

def bgr_to_gray(image):
    gray_image = (0.299 * image[..., 2] + 
                  0.587 * image[..., 1] + 
                  0.114 * image[..., 0])
    return gray_image.astype(np.uint8)

def cvtColor(image, code):
    if code == 'COLOR_BGR2HSV':
        return bgr_to_hsv(image)
    elif code == 'COLOR_BGR2GRAY':
        return bgr_to_gray(image)
    else:
        raise ValueError("Unsupported color conversion code")
    
def drawContours(image, contours, contourIdx, color, thickness):
    """
    Draw contours on an image.

    Parameters:
    - image: NumPy array of shape (H, W, 3) representing the BGR image.
    - contours: List of NumPy arrays, each containing (x, y) coordinates of contour points.
    - contourIdx: Index of the contour to draw. If -1, all contours are drawn.
    - color: Tuple of (B, G, R) values.
    - thickness: Thickness of the contour lines.
    """
    if contourIdx == -1:
        selected_contours = contours
    else:
        if contourIdx < 0 or contourIdx >= len(contours):
            print(f"Invalid contourIdx: {contourIdx}. No contours drawn.")
            return
        selected_contours = [contours[contourIdx]]

    for idx, contour in enumerate(selected_contours):
        print(f"Drawing contour {idx + 1}/{len(selected_contours)} with {len(contour)} points.")
        for i in range(len(contour) - 1):
            pt1 = tuple(contour[i])
            pt2 = tuple(contour[i + 1])
            if len(pt1) != 2 or len(pt2) != 2:
                print(f"Invalid contour points: pt1={pt1}, pt2={pt2}. Skipping this line.")
                continue
            drawLine(image, pt1, pt2, color, thickness)
        # Optionally, connect the last point to the first point to close the contour
        if len(contour) > 1:
            pt1 = tuple(contour[-1])
            pt2 = tuple(contour[0])
            if len(pt1) == 2 and len(pt2) == 2:
                drawLine(image, pt1, pt2, color, thickness)
            else:
                print(f"Invalid closing points: pt1={pt1}, pt2={pt2}. Skipping closure.")

def drawLine(image, pt1, pt2, color, thickness):
    """
    Draw a line on the image from pt1 to pt2 with the given color and thickness.

    Parameters:
    - image: NumPy array of shape (H, W, 3) representing the BGR image.
    - pt1: Tuple of (x, y) coordinates for the start point.
    - pt2: Tuple of (x, y) coordinates for the end point.
    - color: Tuple of (B, G, R) values.
    - thickness: Thickness of the line.
    """
    if not (isinstance(pt1, tuple) and isinstance(pt2, tuple)):
        print(f"Invalid points: pt1={pt1}, pt2={pt2}. Both should be tuples.")
        return

    if len(pt1) != 2 or len(pt2) != 2:
        print(f"Points do not have two elements: pt1={pt1}, pt2={pt2}.")
        return

    x0, y0 = pt1
    x1, y1 = pt2

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0

    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            drawCircle(image, x, y, thickness, color)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            drawCircle(image, x, y, thickness, color)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    # Draw the last point
    drawCircle(image, x, y, thickness, color)

def drawCircle(image, x, y, radius, color):
    """
    Draw a filled circle on the image.

    Parameters:
    - image: NumPy array of shape (H, W, 3) representing the BGR image.
    - x, y: Center coordinates of the circle.
    - radius: Radius of the circle.
    - color: Tuple of (B, G, R) values.
    """
    height, width, _ = image.shape
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                xi, yj = x + i, y + j
                if 0 <= xi < width and 0 <= yj < height:
                    image[yj, xi] = color
