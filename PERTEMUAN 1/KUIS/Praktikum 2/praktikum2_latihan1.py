# ============================================
# PRAKTIKUM 2 - LATIHAN 1: ANALISIS CITRA PRIBADI
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# ---------------- FUNGSI DOWNLOAD SAMPLE ----------------
def download_sample_image():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------------- FUNGSI ANALISIS CITRA PRIBADI ----------------
def analyze_my_image(image_path):
    """Analyze your own image"""
    img = cv2.imread(image_path)
    
    if img is None:
        print("❌ Gambar tidak ditemukan! Periksa path file.")
        return None
    
    print("\n=== ANALISIS CITRA PRIBADI ===")
    
    # 1. DIMENSI DAN RESOLUSI
    height, width, channels = img.shape
    resolution = width * height
    
    print("\n1) DIMENSI DAN RESOLUSI")
    print(f"Width  : {width} pixel")
    print(f"Height : {height} pixel")
    print(f"Channels: {channels}")
    print(f"Resolusi: {resolution:,} pixel")
    
    # 2. ASPECT RATIO
    aspect_ratio = width / height
    print("\n2) ASPECT RATIO")
    print(f"Aspect Ratio = {aspect_ratio:.2f} ({width}:{height})")
    
    # 3. KONVERSI KE GRAYSCALE & PERBANDINGAN UKURAN
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    size_color = img.size * img.dtype.itemsize
    size_gray = gray.size * gray.dtype.itemsize
    
    print("\n3) PERBANDINGAN UKURAN FILE DALAM MEMORI")
    print(f"Ukuran citra warna   : {size_color:,} bytes")
    print(f"Ukuran citra grayscale: {size_gray:,} bytes")
    
    # 4. STATISTIK CITRA
    print("\n4) STATISTIK CITRA GRAYSCALE")
    print(f"Min Intensity : {gray.min()}")
    print(f"Max Intensity : {gray.max()}")
    print(f"Mean Intensity: {gray.mean():.2f}")
    print(f"Std Deviation : {gray.std():.2f}")
    
    # 5. HISTOGRAM SEMUA CHANNEL
    print("\n5) MENAMPILKAN HISTOGRAM...")
    
    plt.figure(figsize=(12,4))
    
    # Histogram grayscale
    plt.subplot(1,2,1)
    plt.hist(gray.ravel(), 256, [0,256], color='gray')
    plt.title("Histogram Grayscale")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    # Histogram warna
    plt.subplot(1,2,2)
    colors = ('b','g','r')
    for i, c in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=c)
    plt.title("Histogram RGB")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend(["Blue", "Green", "Red"])
    
    plt.tight_layout()
    plt.show()
    
    # 6. PERBANDINGAN DENGAN CITRA SAMPLE
    print("\n6) PERBANDINGAN DENGAN CITRA SAMPLE (LENA)")
    sample_img = download_sample_image()
    sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    
    print("\n--- Citra Pribadi ---")
    print(f"Resolusi: {resolution}")
    print(f"Mean Intensity: {gray.mean():.2f}")
    
    print("\n--- Citra Sample ---")
    print(f"Resolusi: {sample_img.shape[0] * sample_img.shape[1]}")
    print(f"Mean Intensity: {sample_gray.mean():.2f}")
    
    # Tampilkan perbandingan gambar
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Citra Pribadi")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title("Citra Sample (Lena)")
    plt.axis('off')
    
    plt.show()
    
    # Return hasil analisis
    analysis_results = {
        "width": width,
        "height": height,
        "channels": channels,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "size_color_bytes": size_color,
        "size_gray_bytes": size_gray,
        "min_intensity": int(gray.min()),
        "max_intensity": int(gray.max()),
        "mean_intensity": float(gray.mean()),
        "std_deviation": float(gray.std())
    }
    
    return analysis_results

# ---------------- MAIN PROGRAM ----------------
if __name__ == "__main__":
    # Ganti dengan path foto kamu
    image_path = "foto_saya.jpg"
    
    results = analyze_my_image(image_path)
    print("\n=== HASIL AKHIR ANALISIS ===")
    print(results)
