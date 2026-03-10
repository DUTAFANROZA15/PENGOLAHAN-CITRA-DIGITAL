import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.measure import shannon_entropy

# ==============================
# MEMBACA CITRA
# ==============================

citra_gelap = cv2.imread("citra_gelap.jpg",0)
citra_terang = cv2.imread("citra_terang.jpg",0)
citra_pencahayaan = cv2.imread("citra_pencahayaan_tidak_merata.jpg",0)

# ==============================
# FUNGSI METRIK
# ==============================

def contrast_ratio(img):
    return img.max() - img.min()

def entropy_value(img):
    return shannon_entropy(img)

# ==============================
# POINT PROCESSING
# ==============================

def negative(img):
    return 255 - img

def log_transform(img):
    c = 255/np.log(1+np.max(img))
    log = c*np.log(1+img)
    return np.array(log,dtype=np.uint8)

def gamma_transform(img,gamma):
    norm = img/255.0
    gamma_img = np.power(norm,gamma)
    return np.uint8(gamma_img*255)

# ==============================
# HISTOGRAM ENHANCEMENT
# ==============================

def contrast_stretch_manual(img):
    rmin = 50
    rmax = 200
    stretched = np.clip((img-rmin)*(255/(rmax-rmin)),0,255)
    return stretched.astype(np.uint8)

def contrast_stretch_auto(img):
    p2,p98 = np.percentile(img,(2,98))
    img_rescale = exposure.rescale_intensity(img,in_range=(p2,p98))
    return img_rescale

def hist_equalization(img):
    return cv2.equalizeHist(img)

def clahe(img):
    c = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    return c.apply(img)

# ==============================
# HISTOGRAM
# ==============================

def tampil_histogram(img,judul):
    plt.figure()
    plt.hist(img.ravel(),256,[0,256])
    plt.title(judul)
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah Pixel")
    plt.show()

# ==============================
# POINT PROCESSING RESULT
# ==============================

negatif = negative(citra_gelap)
log_img = log_transform(citra_gelap)

gamma1 = gamma_transform(citra_gelap,0.5)
gamma2 = gamma_transform(citra_gelap,1.5)
gamma3 = gamma_transform(citra_gelap,2.5)

# ==============================
# HISTOGRAM ENHANCEMENT RESULT
# ==============================

cs_manual = contrast_stretch_manual(citra_pencahayaan)
cs_auto = contrast_stretch_auto(citra_pencahayaan)
he = hist_equalization(citra_terang)
clahe_img = clahe(citra_pencahayaan)

# ==============================
# HALAMAN 1 : DATASET CITRA
# ==============================

plt.figure(figsize=(10,6))
plt.suptitle("Halaman 1 : Dataset Citra Awal")

plt.subplot(1,3,1)
plt.imshow(citra_gelap,cmap='gray')
plt.title("Citra Gelap (Underexposed)")
plt.xlabel("Citra terlalu gelap karena kekurangan cahaya")

plt.subplot(1,3,2)
plt.imshow(citra_terang,cmap='gray')
plt.title("Citra Terang (Overexposed)")
plt.xlabel("Citra terlalu terang karena kelebihan cahaya")

plt.subplot(1,3,3)
plt.imshow(citra_pencahayaan,cmap='gray')
plt.title("Iluminasi Tidak Merata")
plt.xlabel("Citra memiliki bayangan atau silau")

plt.tight_layout()
plt.show()

# ==============================
# HALAMAN 2 : POINT PROCESSING
# ==============================

plt.figure(figsize=(10,8))
plt.suptitle("Halaman 2 : Peningkatan Citra dengan Point Processing")

plt.subplot(2,3,1)
plt.imshow(negatif,cmap='gray')
plt.title("Negative Transformation")
plt.xlabel("Transformasi negatif citra")

plt.subplot(2,3,2)
plt.imshow(log_img,cmap='gray')
plt.title("Log Transformation")
plt.xlabel("Meningkatkan detail pada area gelap")

plt.subplot(2,3,3)
plt.imshow(gamma1,cmap='gray')
plt.title("Gamma = 0.5")
plt.xlabel("Mencerahkan citra")

plt.subplot(2,3,4)
plt.imshow(gamma2,cmap='gray')
plt.title("Gamma = 1.5")
plt.xlabel("Penyesuaian kontras sedang")

plt.subplot(2,3,5)
plt.imshow(gamma3,cmap='gray')
plt.title("Gamma = 2.5")
plt.xlabel("Menggelapkan citra")

plt.tight_layout()
plt.show()

# ==============================
# HALAMAN 3 : HISTOGRAM BASED
# ==============================

plt.figure(figsize=(10,8))
plt.suptitle("Halaman 3 : Peningkatan Citra Berbasis Histogram")

plt.subplot(2,2,1)
plt.imshow(cs_manual,cmap='gray')
plt.title("Contrast Stretching Manual")
plt.xlabel("Peningkatan kontras dengan batas intensitas manual")

plt.subplot(2,2,2)
plt.imshow(cs_auto,cmap='gray')
plt.title("Contrast Stretching Otomatis")
plt.xlabel("Rentang intensitas disesuaikan otomatis")

plt.subplot(2,2,3)
plt.imshow(he,cmap='gray')
plt.title("Histogram Equalization")
plt.xlabel("Redistribusi intensitas secara global")

plt.subplot(2,2,4)
plt.imshow(clahe_img,cmap='gray')
plt.title("CLAHE (Adaptive Histogram Equalization)")
plt.xlabel("Peningkatan kontras secara lokal")

plt.tight_layout()
plt.show()

# ==============================
# HALAMAN 4 : HISTOGRAM
# ==============================

tampil_histogram(citra_gelap,"Histogram Citra Gelap Sebelum Enhancement")
tampil_histogram(log_img,"Histogram Setelah Log Transformation")

tampil_histogram(citra_terang,"Histogram Citra Terang Sebelum Enhancement")
tampil_histogram(he,"Histogram Setelah Histogram Equalization")

# ==============================
# METRIK EVALUASI
# ==============================

print("===== METRIK EVALUASI =====")

print("Contrast Citra Gelap:",contrast_ratio(citra_gelap))
print("Contrast Setelah Log:",contrast_ratio(log_img))

print("Entropy Citra Gelap:",entropy_value(citra_gelap))
print("Entropy Setelah Log:",entropy_value(log_img))

print("Contrast Citra Terang:",contrast_ratio(citra_terang))
print("Contrast Setelah HE:",contrast_ratio(he))

print("Entropy Citra Terang:",entropy_value(citra_terang))
print("Entropy Setelah HE:",entropy_value(he))
