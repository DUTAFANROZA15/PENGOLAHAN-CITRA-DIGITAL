"""
ANALISIS DAN FILTERING DOMAIN FREKUENSI DENGAN FFT DAN WAVELET (FINAL)

Urutan:
1. Citra input
2. FFT (Magnitude & Phase)
3. Rekonstruksi FFT
4. Frekuensi dominan
5. Filtering (Lowpass & Highpass)
6. Variasi cutoff
7. Notch filter
8. Wavelet (koefisien)
9. Wavelet rekonstruksi
10. Perbandingan spasial vs frekuensi
11. Evaluasi (PSNR & waktu)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import time


# ==========================================================
# MEMBACA CITRA
# ==========================================================

img1 = cv2.imread("natural.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("noise_periodic.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Gambar tidak ditemukan")
    exit()

img1 = cv2.resize(img1, (256,256))
img2 = cv2.resize(img2, (256,256))


# ==========================================================
# FFT
# ==========================================================

def fft_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift)+1)
    phase = np.angle(fshift)
    return fshift, magnitude, phase


def reconstruct_phase_only(phase):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(np.exp(1j*phase))))


def reconstruct_magnitude_only(magnitude):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude)))


# ==========================================================
# FILTER
# ==========================================================

def ideal_lowpass(shape, cutoff):
    mask = np.zeros(shape)
    c = shape[0]//2
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i-c)**2 + (j-c)**2 <= cutoff**2:
                mask[i,j] = 1
    return mask


def gaussian_lowpass(shape, cutoff):
    mask = np.zeros(shape)
    c = shape[0]//2
    for i in range(shape[0]):
        for j in range(shape[1]):
            d = (i-c)**2 + (j-c)**2
            mask[i,j] = np.exp(-d/(2*cutoff**2))
    return mask


def ideal_highpass(shape, cutoff):
    return 1 - ideal_lowpass(shape, cutoff)


def gaussian_highpass(shape, cutoff):
    return 1 - gaussian_lowpass(shape, cutoff)


def apply_filter(img, mask):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    return np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))


# ==========================================================
# NOTCH
# ==========================================================

def notch_filter(shape, points, radius=5):
    mask = np.ones(shape)
    for (x,y) in points:
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i-x)**2 + (j-y)**2 <= radius**2:
                    mask[i,j] = 0
    return mask


# ==========================================================
# WAVELET
# ==========================================================

def wavelet_process(img):
    coeffs = pywt.wavedec2(img, 'db4', level=2)
    cA, (cH, cV, cD), (cH2, cV2, cD2) = coeffs

    coeffs_mod = [
        cA,
        (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)),
        (np.zeros_like(cH2), np.zeros_like(cV2), np.zeros_like(cD2))
    ]

    rec = pywt.waverec2(coeffs_mod, 'db4')
    rec = rec[:img.shape[0], :img.shape[1]]
    return np.clip(rec,0,255)


# ==========================================================
# METRIK
# ==========================================================

def mse(a,b):
    return np.mean((a-b)**2)

def psnr(a,b):
    m = mse(a,b)
    return 20*np.log10(255.0/np.sqrt(m)) if m!=0 else 100


# ==========================================================
# PROSES FFT
# ==========================================================

_, mag1, phase1 = fft_spectrum(img1)
rec_phase = reconstruct_phase_only(phase1)
rec_mag = reconstruct_magnitude_only(mag1)


# ==========================================================
# FILTERING
# ==========================================================

img_ideal = apply_filter(img1, ideal_lowpass(img1.shape,30))
img_gauss = apply_filter(img1, gaussian_lowpass(img1.shape,30))
img_hp_ideal = apply_filter(img1, ideal_highpass(img1.shape,30))
img_hp_gauss = apply_filter(img1, gaussian_highpass(img1.shape,30))

notch = notch_filter(img2.shape, [(128,100),(128,156)])
img_notch = apply_filter(img2, notch)


# ==========================================================
# WAVELET
# ==========================================================

coeffs_vis = pywt.wavedec2(img1, 'db4', level=2)
cA, (cH, cV, cD), _ = coeffs_vis

wavelet_rec = wavelet_process(img1)


# ==========================================================
# VISUALISASI SESUAI URUTAN
# ==========================================================

# 1. INPUT
plt.figure(figsize=(8,4))
plt.suptitle("Citra Input")
plt.subplot(1,2,1); plt.imshow(img1,cmap='gray'); plt.title("Natural"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(img2,cmap='gray'); plt.title("Noise"); plt.axis('off')
plt.show()

# 2. FFT
plt.figure(figsize=(12,4))
plt.suptitle("FFT")
plt.subplot(1,3,1); plt.imshow(img1,cmap='gray'); plt.title("Asli")
plt.subplot(1,3,2); plt.imshow(mag1,cmap='gray'); plt.title("Magnitude")
plt.subplot(1,3,3); plt.imshow(phase1,cmap='gray'); plt.title("Phase")
plt.show()

# 3. REKONSTRUKSI
plt.figure(figsize=(10,4))
plt.suptitle("Rekonstruksi FFT")
plt.subplot(1,3,1); plt.imshow(img1,cmap='gray'); plt.title("Asli")
plt.subplot(1,3,2); plt.imshow(rec_phase,cmap='gray'); plt.title("Phase Only")
plt.subplot(1,3,3); plt.imshow(rec_mag,cmap='gray'); plt.title("Magnitude Only")
plt.show()

# 4. FREKUENSI DOMINAN
plt.figure(figsize=(5,5))
plt.title("Frekuensi Dominan")
plt.imshow(mag1,cmap='gray')
plt.axis('off')
plt.show()

# 5. FILTER LP & HP
plt.figure(figsize=(12,4))
plt.suptitle("Filtering")
plt.subplot(1,4,1); plt.imshow(img_ideal,cmap='gray'); plt.title("Ideal LP")
plt.subplot(1,4,2); plt.imshow(img_gauss,cmap='gray'); plt.title("Gaussian LP")
plt.subplot(1,4,3); plt.imshow(img_hp_ideal,cmap='gray'); plt.title("Ideal HP")
plt.subplot(1,4,4); plt.imshow(img_hp_gauss,cmap='gray'); plt.title("Gaussian HP")
plt.show()

# 6. VARIASI CUTOFF
plt.figure(figsize=(10,6))
plt.suptitle("Cutoff Effect")
for i,c in enumerate([10,30,60]):
    res = apply_filter(img1, gaussian_lowpass(img1.shape,c))
    plt.subplot(2,2,i+1)
    plt.title(f"Cutoff {c}")
    plt.imshow(res,cmap='gray')
    plt.axis('off')
plt.show()

# 7. NOTCH
plt.figure(figsize=(10,4))
plt.suptitle("Notch Filter")
plt.subplot(1,2,1); plt.imshow(img2,cmap='gray'); plt.title("Sebelum")
plt.subplot(1,2,2); plt.imshow(img_notch,cmap='gray'); plt.title("Sesudah")
plt.show()

# 8. WAVELET KOEFISIEN
plt.figure(figsize=(10,4))
plt.suptitle("Koefisien Wavelet")
plt.subplot(1,4,1); plt.imshow(cA,cmap='gray'); plt.title("cA")
plt.subplot(1,4,2); plt.imshow(cH,cmap='gray'); plt.title("cH")
plt.subplot(1,4,3); plt.imshow(cV,cmap='gray'); plt.title("cV")
plt.subplot(1,4,4); plt.imshow(cD,cmap='gray'); plt.title("cD")
plt.show()

# 9. WAVELET RESULT
plt.figure(figsize=(8,4))
plt.suptitle("Wavelet")
plt.subplot(1,2,1); plt.imshow(img1,cmap='gray'); plt.title("Asli")
plt.subplot(1,2,2); plt.imshow(wavelet_rec,cmap='gray'); plt.title("Hasil")
plt.show()

# 10. SPASIAL VS FREKUENSI
spatial = cv2.GaussianBlur(img1,(5,5),0)

plt.figure(figsize=(10,4))
plt.suptitle("Spasial vs Frekuensi")
plt.subplot(1,3,1); plt.imshow(img1,cmap='gray'); plt.title("Asli")
plt.subplot(1,3,2); plt.imshow(spatial,cmap='gray'); plt.title("Spasial")
plt.subplot(1,3,3); plt.imshow(img_gauss,cmap='gray'); plt.title("Frekuensi")
plt.show()


# ==========================================================
# EVALUASI
# ==========================================================

print("\nHASIL EVALUASI")
print("="*50)

start = time.perf_counter()
p1 = psnr(img1, img_ideal)
t1 = time.perf_counter() - start

start = time.perf_counter()
p2 = psnr(img1, img_gauss)
t2 = time.perf_counter() - start

start = time.perf_counter()
p3 = psnr(img1, wavelet_rec)
t3 = time.perf_counter() - start

print(f"Ideal LP     : {p1:.2f} | Time {t1:.8f}")
print(f"Gaussian LP  : {p2:.2f} | Time {t2:.8f}")
print(f"Wavelet      : {p3:.2f} | Time {t3:.8f}")
