#============================================
#PRAKTIKUM 2: ANALISIS MODEL WARNA & ALIASING
#============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== PRAKTIKUM 2: ANALISIS MODEL WARNA & ALIASING ===")

# ==========================================================
# LOAD 2 GAMBAR
# ==========================================================

image1 = cv2.imread("image1.jpg")  # Wajah / objek dengan detail
image2 = cv2.imread("image2.jpg")  # Dokumen / bayangan

# Jika gambar tidak ditemukan → buat sintetik
if image1 is None:
    image1 = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(image1, (200,200), 100, (45, 34, 200), -1)
    cv2.putText(image1, "PATTERN", (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255,255), 2)
    print("Menggunakan image1 sintetik")

if image2 is None:
    image2 = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.putText(image2, "DIGITAL IMAGE", (30,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255,255,255), 3)
    print("Menggunakan image2 sintetik")

# ==========================================================
# 1. ANALISIS MODEL WARNA
# ==========================================================

def skin_detection_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result


def shadow_removal_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    L_enhanced = clahe.apply(L)
    merged = cv2.merge([L_enhanced, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def text_extraction_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ==========================================================
# 2. SIMULASI ALIASING (PAKAI IMAGE1)
# ==========================================================

def simulate_aliasing(image, factors):
    results = []
    for factor in factors:
        small = cv2.resize(image,
                           (image.shape[1]//factor,
                            image.shape[0]//factor),
                           interpolation=cv2.INTER_NEAREST)

        restored = cv2.resize(small,
                              (image.shape[1],
                               image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        results.append((factor, restored))
    return results


# ==========================================================
# PROSES GAMBAR 1 (Skin + Aliasing)
# ==========================================================

skin_mask, skin_result = skin_detection_hsv(image1)

aliasing_results = simulate_aliasing(image1, [2,4,8])

# ==========================================================
# PROSES GAMBAR 2 (Shadow + Text)
# ==========================================================

shadow_result = shadow_removal_lab(image2)
text_result = text_extraction_gray(image2)

# ==========================================================
# VISUALISASI
# ==========================================================

plt.figure(figsize=(15,10))

# IMAGE 1
plt.subplot(3,3,1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title("Image 1 (Original)")
plt.axis("off")

plt.subplot(3,3,2)
plt.imshow(skin_mask, cmap='gray')
plt.title("Skin Mask (HSV)")
plt.axis("off")

plt.subplot(3,3,3)
plt.imshow(cv2.cvtColor(skin_result, cv2.COLOR_BGR2RGB))
plt.title("Skin Detection Result")
plt.axis("off")

# Aliasing
for i, (factor, img_alias) in enumerate(aliasing_results):
    plt.subplot(3,3,4+i)
    plt.imshow(cv2.cvtColor(img_alias, cv2.COLOR_BGR2RGB))
    plt.title(f"Aliasing Factor {factor}")
    plt.axis("off")

# IMAGE 2
plt.subplot(3,3,7)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Image 2 (Original)")
plt.axis("off")

plt.subplot(3,3,8)
plt.imshow(cv2.cvtColor(shadow_result, cv2.COLOR_BGR2RGB))
plt.title("Shadow Removal (LAB)")
plt.axis("off")

plt.subplot(3,3,9)
plt.imshow(text_result, cmap='gray')
plt.title("Text Extraction (Grayscale)")
plt.axis("off")

plt.suptitle("Analisis Model Warna & Simulasi Aliasing")
plt.tight_layout()
plt.show()

print("\n=== PRAKTIKUM SELESAI ===")
print("Image 1 → Skin Detection + Aliasing")
print("Image 2 → Shadow Removal + Text Extraction")
