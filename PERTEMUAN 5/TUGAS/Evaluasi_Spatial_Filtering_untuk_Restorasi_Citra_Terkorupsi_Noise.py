"""
EVALUASI SPATIAL FILTERING UNTUK RESTORASI CITRA TERKORUPSI NOISE

Program ini melakukan:
1. Membaca citra asli
2. Membuat 3 jenis noise (Gaussian, Salt & Pepper, Speckle)
3. Mengimplementasikan filter linear dan non-linear
4. Menghitung metrik evaluasi (MSE, PSNR, SSIM)
5. Mengukur waktu komputasi
6. Menampilkan tabel hasil evaluasi di terminal
7. Menampilkan visualisasi untuk inspeksi visual
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim


# ==========================================================
# MEMBACA CITRA ASLI
# ==========================================================
# Membaca citra yang akan digunakan sebagai referensi

image = cv2.imread("kucing.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Gambar tidak ditemukan")
    exit()

# Resize agar ukuran seragam
image = cv2.resize(image, (256,256))


# ==========================================================
# FUNGSI MENAMBAHKAN NOISE
# ==========================================================

# Gaussian Noise
def add_gaussian_noise(img):

    mean = 0
    std = 25

    gaussian = np.random.normal(mean, std, img.shape)

    noisy = img + gaussian
    noisy = np.clip(noisy,0,255)

    return noisy.astype(np.uint8)


# Salt and Pepper Noise
def add_sp_noise(img):

    noisy = img.copy()
    prob = 0.02

    rnd = np.random.rand(*img.shape)

    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255

    return noisy


# Speckle Noise
def add_speckle_noise(img):

    noise = np.random.randn(*img.shape)

    noisy = img + img * noise * 0.2

    noisy = np.clip(noisy,0,255)

    return noisy.astype(np.uint8)



# ==========================================================
# MEMBUAT VARIASI NOISE
# ==========================================================

gaussian_noise = add_gaussian_noise(image)
sp_noise = add_sp_noise(image)
speckle_noise = add_speckle_noise(image)



# ==========================================================
# FUNGSI METRIK EVALUASI
# ==========================================================

def mse(original, restored):

    return np.mean((original - restored) ** 2)



def psnr(original, restored):

    mse_val = mse(original, restored)

    if mse_val == 0:
        return 100

    PIXEL_MAX = 255.0

    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))



def ssim_value(original, restored):

    return ssim(original, restored)



# ==========================================================
# DEFINISI FILTER
# ==========================================================

filters = {

    "Mean_3x3": lambda img: cv2.blur(img,(3,3)),
    "Mean_7x7": lambda img: cv2.blur(img,(7,7)),

    "Gaussian_sigma1": lambda img: cv2.GaussianBlur(img,(5,5),1),
    "Gaussian_sigma2": lambda img: cv2.GaussianBlur(img,(5,5),2),

    "Median_3x3": lambda img: cv2.medianBlur(img,3),
    "Median_7x7": lambda img: cv2.medianBlur(img,7),

    "Min_Filter": lambda img: cv2.erode(img,np.ones((3,3),np.uint8))
}



# ==========================================================
# MENYIMPAN SEMUA NOISE DALAM DICTIONARY
# ==========================================================

noise_images = {

    "Gaussian Noise": gaussian_noise,
    "Salt & Pepper Noise": sp_noise,
    "Speckle Noise": speckle_noise
}



# ==========================================================
# EVALUASI SEMUA FILTER PADA SEMUA NOISE
# ==========================================================

results = []

for noise_name, noisy_img in noise_images.items():

    for filter_name, filter_func in filters.items():

        start = time.time()

        restored = filter_func(noisy_img)

        end = time.time()

        comp_time = end - start

        mse_val = mse(image, restored)
        psnr_val = psnr(image, restored)
        ssim_val = ssim_value(image, restored)

        results.append([noise_name, filter_name, mse_val, psnr_val, ssim_val, comp_time])



# ==========================================================
# MENAMPILKAN TABEL HASIL DI TERMINAL
# ==========================================================

print("\nTABEL PERBANDINGAN METRIK FILTER")
print("="*90)

print(f"{'Noise':20} {'Filter':20} {'MSE':10} {'PSNR':10} {'SSIM':10} {'Time(s)':10}")

print("-"*90)

for r in results:

    print(f"{r[0]:20} {r[1]:20} {r[2]:10.2f} {r[3]:10.2f} {r[4]:10.3f} {r[5]:10.5f}")



# ==========================================================
# VISUAL INSPECTION
# ==========================================================

# ----------------------------------------------------------
# CITRA ASLI DAN VARIASI KORUPSI NOISE (DIGABUNG)
# ----------------------------------------------------------

plt.figure(figsize=(12,5))
plt.suptitle("Citra Asli dan Variasi Korupsi Noise")

plt.subplot(1,4,1)
plt.title("Citra Asli")
plt.imshow(image,cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Gaussian Noise")
plt.imshow(gaussian_noise,cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Salt and Pepper Noise")
plt.imshow(sp_noise,cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("Speckle Noise")
plt.imshow(speckle_noise,cmap='gray')
plt.axis('off')

plt.show()



# ----------------------------------------------------------
# IMPLEMENTASI FILTER LINEAR
# ----------------------------------------------------------

plt.figure(figsize=(10,5))
plt.suptitle("Implementasi Filter Linear")

plt.subplot(2,2,1)
plt.title("Mean Filter 3x3")
plt.imshow(filters["Mean_3x3"](gaussian_noise),cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Mean Filter 7x7")
plt.imshow(filters["Mean_7x7"](gaussian_noise),cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Gaussian Filter Sigma 1")
plt.imshow(filters["Gaussian_sigma1"](gaussian_noise),cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Gaussian Filter Sigma 2")
plt.imshow(filters["Gaussian_sigma2"](gaussian_noise),cmap='gray')
plt.axis('off')

plt.show()



# ----------------------------------------------------------
# IMPLEMENTASI FILTER NON LINEAR
# ----------------------------------------------------------

plt.figure(figsize=(10,5))
plt.suptitle("Implementasi Filter Non Linear")

plt.subplot(2,2,1)
plt.title("Median Filter 3x3")
plt.imshow(filters["Median_3x3"](sp_noise),cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Median Filter 7x7")
plt.imshow(filters["Median_7x7"](sp_noise),cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Min Filter")
plt.imshow(filters["Min_Filter"](sp_noise),cmap='gray')
plt.axis('off')

plt.show()



# ==========================================================
# ANALISIS FILTER TERBAIK BERDASARKAN PSNR
# ==========================================================

print("\nANALISIS FILTER TERBAIK BERDASARKAN PSNR")
print("="*60)

for noise_name in noise_images.keys():

    subset = [r for r in results if r[0] == noise_name]

    best = max(subset, key=lambda x: x[3])

    print(f"{noise_name} -> Filter terbaik: {best[1]} (PSNR = {best[3]:.2f})")
