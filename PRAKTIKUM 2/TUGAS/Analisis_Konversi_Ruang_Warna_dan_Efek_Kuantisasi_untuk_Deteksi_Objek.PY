# ==========================================================
# ANALISIS KONVERSI RUANG WARNA DAN EFEK KUANTISASI
# File gambar: terang.jpg, normal.jpg, gelap.jpg
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================================
# LOAD 3 CITRA
# ==========================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: File {path} tidak ditemukan!")
        exit()
    return img

images = {
    "Terang": load_image("terang.jpg"),
    "Normal": load_image("normal.jpg"),
    "Gelap": load_image("gelap.jpg")
}

# ==========================================================
# 1. konversi ruang warna menggunakan OpenCV dan NumPy:
# ==========================================================

def convert_color_spaces(image):
    start = time.time()

    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    end = time.time()
    return rgb, gray, hsv, lab, (end - start)

# ==========================================================
# 2. Implementasikan teknik kuantisasi pada masing-masing model warna:
# ==========================================================

def uniform_quantization(image, levels=16):
    step = 256 // levels
    return ((image // step) * step).astype(np.uint8)

def non_uniform_quantization(image, k=16):
    Z = image.reshape((-1,1)) if len(image.shape)==2 else image.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(image.shape)

# ==========================================================
# 3. ANALISIS DAN VISUALISASI EFEK KUANTISASI
# ==========================================================

def show_histogram(image, title):
    if len(image.shape) == 2:
        plt.hist(image.ravel(), bins=256)
    else:
        for i in range(3):
            plt.hist(image[:,:,i].ravel(), bins=256, alpha=0.5)
    plt.title(title)

def segmentasi_threshold(gray):
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return thresh

# ==========================================================
# 4. ANALISIS PARAMETER CITRA
# ==========================================================

def ukuran_memori(image):
    return image.nbytes

def rasio_kompresi(original, hasil):
    return original.nbytes / hasil.nbytes

# ==========================================================
# PROSES UTAMA
# ==========================================================

for nama, img in images.items():

    print("\n==================================================")
    print(f"ANALISIS CITRA: {nama}")
    print("==================================================")

    height, width, channel = img.shape
    print("\n2. REPRESENTASI DIGITAL CITRA")
    print("Resolusi:", width, "x", height)
    print("Jumlah Channel:", channel)
    print("Shape Array:", img.shape)

    # ===============================
    # 2.2.1 REPRESENTASI MATRIKS
    # ===============================
    print("\n2.2.1 REPRESENTASI MATRIKS (5x5 piksel pertama):")
    print(img[:5, :5])

    # ===============================
    # 2.2.2 REPRESENTASI VEKTOR
    # ===============================
    print("\n2.2.2 REPRESENTASI VEKTOR (25 nilai pertama hasil flatten):")
    vector = img.flatten()
    print(vector[:25])

    # ======================================================
    # KONVERSI WARNA
    # ======================================================

    rgb, gray, hsv, lab, waktu = convert_color_spaces(img)

    # ======================================================
    # KUANTISASI
    # ======================================================

    gray_u = uniform_quantization(gray)
    gray_nu = non_uniform_quantization(gray)

    # ======================================================
    # SEGMENTASI
    # ======================================================

    seg_asli = segmentasi_threshold(gray)
    seg_u = segmentasi_threshold(gray_u)
    seg_nu = segmentasi_threshold(gray_nu)

    # ======================================================
    # OUTPUT VISUAL
    # ======================================================

    plt.figure(figsize=(12,8))
    plt.suptitle(f"1. KONVERSI RUANG WARNA - {nama}", fontsize=14)

    plt.subplot(2,2,1)
    plt.imshow(rgb)
    plt.title("RGB")

    plt.subplot(2,2,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")

    plt.subplot(2,2,3)
    plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title("HSV")

    plt.subplot(2,2,4)
    plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    plt.title("LAB")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    plt.suptitle(f"2. HASIL KUANTISASI - {nama}", fontsize=14)

    plt.subplot(1,3,1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(gray_u, cmap='gray')
    plt.title("Uniform (16 Level)")

    plt.subplot(1,3,3)
    plt.imshow(gray_nu, cmap='gray')
    plt.title("Non-Uniform")

    plt.tight_layout()
    plt.show()

    # ======================================================
    # PARAMETER TEKNIS
    # ======================================================

    print("\n4. ANALISIS PARAMETER CITRA")
    print("Waktu konversi warna:", round(waktu,6), "detik")
    print("Memori asli:", ukuran_memori(img), "bytes")
    print("Memori grayscale:", ukuran_memori(gray), "bytes")
    print("Rasio kompresi:", round(rasio_kompresi(img, gray),2))
