import numpy as np

def dilate(image, kernel, iterations=1):
    np_image = np.array(image)
    if np_image.ndim == 2:  
        np_image = np_image[:, :, np.newaxis]
    
    kernel = np.array(kernel)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(np_image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    dilated_image = np.zeros_like(np_image)
    
    for _ in range(iterations):
        for i in range(pad, padded_image.shape[0] - pad):
            for j in range(pad, padded_image.shape[1] - pad):
                for k in range(np_image.shape[2]): 
                    region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1, k]
                    dilated_image[i - pad, j - pad, k] = np.max(region * kernel)
        padded_image = np.pad(dilated_image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    if dilated_image.shape[2] == 1: 
        dilated_image = dilated_image[:, :, 0]
    
    return dilated_image

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

def bgr_to_hsv(image):
    image = image.astype('float32') / 255.0
    hsv_image = np.zeros_like(image)
    
    b, g, r = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=-1)
    minc = np.min(image, axis=-1)
    delta = maxc - minc

    s = np.zeros_like(maxc)
    mask = maxc != 0
    s[mask] = delta[mask] / maxc[mask]
    
    h = np.zeros_like(maxc)
    
    mask_delta = delta != 0
    mask_r = (maxc == r) & mask_delta
    mask_g = (maxc == g) & mask_delta
    mask_b = (maxc == b) & mask_delta

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (2.0 + (b[mask_g] - r[mask_g]) / delta[mask_g])
    h[mask_b] = (4.0 + (r[mask_b] - g[mask_b]) / delta[mask_b])

    h = (h / 6.0) * 180.0

    s = s * 255.0
    v = maxc * 255.0

    hsv_image[..., 0] = h
    hsv_image[..., 1] = s
    hsv_image[..., 2] = v

    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)
    
    return hsv_image

def bgr_to_gray(image):
    gray_image = (0.299 * image[..., 0] + 
                  0.587 * image[..., 1] + 0.114 * image[..., 2] 
                  )
    return gray_image.astype(np.uint8)

def cvtColor(image, code):
    if code == 'COLOR_BGR2HSV':
        return bgr_to_hsv(image)
    elif code == 'COLOR_BGR2GRAY':
        return bgr_to_gray(image)
    else:
        raise ValueError("Unsupported color conversion code")

def gaussian_blur(src, ksize=5, sigma=1.4):

    k = (ksize - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    g /= g.sum()
    return filter2D(src, g)

def filter2D(src, kernel):
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_src = np.pad(src, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    dst = np.zeros_like(src, dtype=np.float32)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            region = padded_src[i:i + k_height, j:j + k_width]
            dst[i, j] = np.sum(region * kernel)

    return dst

def sobel_gradients(src):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    Ix = filter2D(src, Kx)
    Iy = filter2D(src, Ky)
    magnitude = np.hypot(Ix, Iy)
    direction = np.arctan2(Iy, Ix)
    return magnitude, direction

def non_max_suppression(mag, direction):
    M, N = mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = np.degrees(direction) % 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q, r = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if mag[i, j] >= q and mag[i, j] >= r:
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0
    return Z

def threshold_edges(img, low_ratio=0.1, high_ratio=0.3):
    high = img.max() * high_ratio
    low = high * low_ratio

    res = np.zeros_like(img, dtype=np.uint8)
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def hysteresis(img, weak=75, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if any(img[i + di, j + dj] == strong
                       for di, dj in [(-1, -1), (-1, 0), (-1, 1),
                                      (0, -1),   (0, 1),
                                      (1, -1),   (1, 0), (1, 1)]):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detection(image, low_ratio=0.1, high_ratio=0.3, sigma=1.4):

    image = image.astype(np.float32)

    blurred = gaussian_blur(image, ksize=5, sigma=sigma)

    mag, direction = sobel_gradients(blurred)
    mag *= 255.0 / (mag.max() + 1e-5) 

    non_max = non_max_suppression(mag, direction)

    thresh, weak, strong = threshold_edges(non_max, low_ratio, high_ratio)

    result = hysteresis(thresh, weak=weak, strong=strong)
    return np.array(result, dtype=np.uint8)
