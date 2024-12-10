import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter

def gaussian_blur(image, sigma=1.0):
    """
    Gaussian Blur uygulayan fonksiyon.
    :param image: Giriş görüntüsü (PIL Image)
    :param sigma: Gaussian Blur sigma değeri
    :return: Blur uygulanmış görüntü (PIL Image)
    """
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))

def median_filter(image, size=3):
    """
    Median filtre uygulayan fonksiyon.
    :param image: Giriş görüntüsü (PIL Image)
    :param size: Filtre boyutu (tek sayı olmalı)
    :return: Filtre uygulanmış görüntü (PIL Image)
    """
    return image.filter(ImageFilter.MedianFilter(size=size))

def compute_hog(image):
    # HOG parametreleri
    cell_size = (8, 8)  # Her hücredeki piksel sayısı
    block_size = (2, 2)  # Her bloktaki hücre sayısı
    nbins = 9  # Gradyan yönlerinin sayısı

    # Görüntü boyutları
    height, width = image.shape

    # Gradyan hesaplama
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # x yönünde gradyan
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # y yönünde gradyan

    # Gradyan büyüklüğü ve yönü
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    # Hücre histogramları
    n_cellsx = int(np.floor(width / cell_size[1]))
    n_cellsy = int(np.floor(height / cell_size[0]))
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, nbins))

    for i in range(n_cellsy):
        for j in range(n_cellsx):
            cell_magnitude = magnitude[i * cell_size[0]:(i + 1) * cell_size[0],
                                       j * cell_size[1]:(j + 1) * cell_size[1]]
            cell_orientation = orientation[i * cell_size[0]:(i + 1) * cell_size[0],
                                           j * cell_size[1]:(j + 1) * cell_size[1]]
            hist, _ = np.histogram(cell_orientation, bins=nbins, range=(0, 180),
                                   weights=cell_magnitude)
            orientation_histogram[i, j, :] = hist

    return orientation_histogram

def visualize_hog(image, hog_features, cell_size=(8, 8)):
    height, width = image.shape
    hog_image = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(hog_image)
    
    n_cellsy, n_cellsx, nbins = hog_features.shape
    
    for i in range(n_cellsy):
        for j in range(n_cellsx):
            cell_gradient = hog_features[i, j]
            cell_total = np.sum(cell_gradient)
            if cell_total > 0:  # Only draw if there's a significant gradient
                center_x = j * cell_size[1] + cell_size[1] // 2
                center_y = i * cell_size[0] + cell_size[0] // 2
                
                # Find dominant gradient direction
                max_bin = np.argmax(cell_gradient)
                angle = max_bin * 180 / nbins
                
                # Draw line in dominant direction
                magnitude = min(cell_total * 0.25, cell_size[0])  # Scale magnitude
                rad = np.deg2rad(angle)
                x1 = int(center_x - magnitude * np.cos(rad))
                y1 = int(center_y - magnitude * np.sin(rad))
                x2 = int(center_x + magnitude * np.cos(rad))
                y2 = int(center_y + magnitude * np.sin(rad))
                
                draw.line([(x1, y1), (x2, y2)], fill=0, width=1)
    
    return np.array(hog_image)

def mark_edges(image, hog_features, cell_size=(8, 8)):
    image_array = np.array(image)
    
    # Eğer görüntü renkliyse, gri tonlamaya dönüştür
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=2)
    
    height, width = image_array.shape
    draw = ImageDraw.Draw(image)
    
    n_cellsy, n_cellsx, nbins = hog_features.shape
    
    # Edge işaretleme matrisi
    edge_marked = np.zeros((n_cellsy, n_cellsx), dtype=bool)
    
    for i in range(n_cellsy):
        for j in range(n_cellsx):
            cell_gradient = hog_features[i, j]
            cell_total = np.sum(cell_gradient)
            if cell_total > 10400:  # Sadece anlamlı bir gradyanın olduğu hücreler
                center_x = j * cell_size[1] + cell_size[1] // 2
                center_y = i * cell_size[0] + cell_size[0] // 2
                
                # Komşu piksellerin edge olup olmadığını kontrol et
                neighbors = [
                    (i-1, j-1), (i-1, j), (i-1, j+1),
                    (i, j-1),           (i, j+1),
                    (i+1, j-1), (i+1, j), (i+1, j+1)
                ]
                edge_count = sum(
                    1 for ni, nj in neighbors 
                    if 0 <= ni < n_cellsy and 0 <= nj < n_cellsx and edge_marked[ni, nj]
                )
                
                # Piksel geçişlerindeki farkı kontrol et
                pixel_value = image_array[center_y, center_x]
                if pixel_value == 0:
                    continue  # Bölme hatalarını önlemek için
                
                neighbor_values = [
                    image_array[ni * cell_size[0] + cell_size[0] // 2, nj * cell_size[1] + cell_size[1] // 2]
                    for ni, nj in neighbors 
                    if 0 <= ni < n_cellsy and 0 <= nj < n_cellsx
                ]
                # Piksel farklarını yüzde olarak hesapla
                intensity_diff_up = [nv >= 1.7 * pixel_value for nv in neighbor_values]
                intensity_diff_down = [nv <= pixel_value / 1.7 for nv in neighbor_values]
                intensity_diff = intensity_diff_up + intensity_diff_down
                
                # Eğer komşu piksellerin en az 1.7 kat farkı varsa ve çok fazla edge yoksa işaretleme
                if edge_count < 3 and any(intensity_diff):
                    # Merkez pikseli sarı nokta ile işaretle
                    draw.ellipse(
                        (center_x - 1, center_y - 1, center_x + 1, center_y + 1), 
                        fill='yellow'
                    )
                    edge_marked[i, j] = True
    
    return image

def main(image_path):
    # Load and convert image to grayscale
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)

    # Apply Gaussian Blur
    blurred_image = gaussian_blur(image, sigma=1.0)
    # Apply Median Filter
    denoised_image = median_filter(blurred_image, size=3)
    denoised_image_array = np.array(denoised_image)

    # Compute HOG
    hog_features = compute_hog(denoised_image_array)

    # Visualize HOG
    hog_image = visualize_hog(denoised_image_array, hog_features)

    # Show HOG image
    Image.fromarray(hog_image).show(title='HOG Image')

    # Mark edges on the original image
    original_image_with_edges = mark_edges(image.convert('RGB'), hog_features)

    # Show and save results
    original_image_with_edges.show(title='Original Image with Edges')
    # original_image_with_edges.save('output.jpg')

    hog_image_pil = Image.fromarray(hog_image)
    # hog_image_pil.save('hog_result.jpg')

if __name__ == "__main__":
    main('input2.jpg')
