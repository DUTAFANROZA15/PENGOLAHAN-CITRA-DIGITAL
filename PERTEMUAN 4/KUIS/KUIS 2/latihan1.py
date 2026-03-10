# ============================================
# LATIHAN 1: MANUAL HISTOGRAM EQUALIZATION
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    
    Parameters:
    image: Input grayscale image (0-255)
    
    Returns:
    equalized_image: Hasil citra setelah equalization
    transform_function: Fungsi transformasi (mapping 0-255)
    """

    # Pastikan image grayscale
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale!")

    # ----------------------------------------
    # 1. Hitung Histogram
    # ----------------------------------------
    histogram = np.zeros(256, dtype=int)

    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    # ----------------------------------------
    # 2. Hitung Cumulative Histogram (CDF)
    # ----------------------------------------
    cumulative_histogram = np.zeros(256, dtype=int)
    cumulative_histogram[0] = histogram[0]

    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i]

    # ----------------------------------------
    # 3. Hitung Transformation Function
    # s = (L-1) * CDF / (M*N)
    # ----------------------------------------
    M, N = image.shape
    total_pixels = M * N
    L = 256

    transform_function = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        transform_function[i] = round((L - 1) * cumulative_histogram[i] / total_pixels)

    # ----------------------------------------
    # 4. Apply Transformation
    # ----------------------------------------
    equalized_image = np.zeros_like(image)

    for i in range(M):
        for j in range(N):
            equalized_image[i, j] = transform_function[image[i, j]]

    # ----------------------------------------
    # 5. Return hasil
    # ----------------------------------------
    return equalized_image, transform_function


# ============================================
# MAIN PROGRAM
# ============================================

if __name__ == "__main__":

    print("=== MANUAL HISTOGRAM EQUALIZATION ===")

    # Load image
    image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    # Proses Histogram Equalization
    equalized_image, transform_function = manual_histogram_equalization(image)

    # =====================================
    # HALAMAN 1 : HASIL CITRA & HISTOGRAM
    # =====================================

    plt.figure(figsize=(12,8))

    # Original Image
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Equalized Image
    plt.subplot(2,2,2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title("Equalized Image")
    plt.axis('off')

    # Histogram Original
    plt.subplot(2,2,3)
    plt.hist(image.ravel(), bins=256, range=[0,256])
    plt.title("Histogram Original")

    # Histogram Equalized
    plt.subplot(2,2,4)
    plt.hist(equalized_image.ravel(), bins=256, range=[0,256])
    plt.title("Histogram Equalized")

    plt.tight_layout()
    plt.show()

    # =====================================
    # HALAMAN 2 : VISUAL TRANSFORMATION FUNCTION
    # =====================================

    plt.figure(figsize=(8,6))

    x = np.arange(256)

    plt.plot(x, transform_function)

    plt.title("Transformation Function (Histogram Equalization)")
    plt.xlabel("Input Intensity Level (r)")
    plt.ylabel("Output Intensity Level (s)")
    plt.grid(True)

    plt.show()

    print("\nProses selesai!")
