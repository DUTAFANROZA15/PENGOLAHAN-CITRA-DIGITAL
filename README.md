# 🖼️ Pengolahan Citra Digital
### Duta Fanroza — NIM: 24343005 — Universitas Negeri Padang

> Repositori ini mendokumentasikan seluruh tugas praktikum mata kuliah **Pengolahan Citra Digital**, mulai dari konsep dasar digitalisasi citra hingga klasifikasi berbasis deep learning. Setiap pertemuan menghasilkan program Python yang mengimplementasikan teknik pengolahan citra secara sistematis.

---

## 👤 Identitas Mahasiswa

| Atribut | Keterangan |
|--------|-----------|
| **Nama** | Duta Fanroza |
| **NIM** | 24343005 |
| **Mata Kuliah** | Pengolahan Citra Digital |
| **Program Studi** | Pendidikan Teknik Informatika |
| **Universitas** | Universitas Negeri Padang |
| **Sesi** | [202523430039](https://smile.unp.ac.id/elearning/course/view.php?id=3430000001290) |

---

## 📁 Struktur Repositori

```
📦 PENGOLAHAN-CITRA-DIGITAL/
├── 📄 README.md
├── 🐍 digitalisasi_citra.py                                    ← Pertemuan 1
├── 🐍 Analisis_Konversi_Ruang_Warna_dan_Efek_Kuantisasi...py   ← Pertemuan 2
├── 🐍 Pipeline_Transformasi_Geometrik...py                     ← Pertemuan 3
├── 🐍 Pipeline_Enhancement_Citra...py                          ← Pertemuan 4
├── 🐍 Evaluasi_Spatial_Filtering...py                          ← Pertemuan 5
├── 🐍 Pipeline_Restorasi_Citra_untuk_Motion_Blur...py          ← Pertemuan 6
├── 🐍 Analisis_dan_Filtering_Domain_Frekuensi...py             ← Pertemuan 7
├── 🐍 Komparasi_Metode_Segmentasi...py                         ← Pertemuan 9
├── 🐍 Pipeline_Morfologi...py                                  ← Pertemuan 10
├── 🐍 shape_analysis_pipeline.py                               ← Pertemuan 11
├── 🐍 Sistem_Pencocokan_Objek_Berbasis_Fitur_Lokal.py          ← Pertemuan 12
├── 🐍 Komparasi_Klasifikasi_KNNvsSVM.py                        ← Pertemuan 13
└── 🐍 Klasifikasi_Citra_dengan_CNN...py                        ← Pertemuan 14
```

---

## 🗺️ Peta Perjalanan Belajar

```
[ Pertemuan 1 ]  Digitalisasi Citra
       ↓
[ Pertemuan 2 ]  Konversi Ruang Warna & Kuantisasi
       ↓
[ Pertemuan 3 ]  Transformasi Geometrik & Registrasi Citra
       ↓
[ Pertemuan 4 ]  Enhancement: Under/Overexposed
       ↓
[ Pertemuan 5 ]  Spatial Filtering & Noise Removal
       ↓
[ Pertemuan 6 ]  Restorasi Motion Blur
       ↓
[ Pertemuan 7 ]  Domain Frekuensi: FFT & Wavelet
       ↓
[ Pertemuan 8 ]  ★ UJIAN TENGAH SEMESTER ★
       ↓
[ Pertemuan 9 ]  Segmentasi Objek
       ↓
[ Pertemuan 10 ] Morfologi & OCR Preprocessing
       ↓
[ Pertemuan 11 ] Shape Analysis & Klasifikasi Bentuk
       ↓
[ Pertemuan 12 ] Pencocokan Objek Berbasis Fitur Lokal
       ↓
[ Pertemuan 13 ] Klasifikasi: KNN vs SVM
       ↓
[ Pertemuan 14 ] CNN: From Scratch → Transfer Learning
```

---

## 📚 Detail Tugas Per Pertemuan

---

### Pertemuan 1 — Eksplorasi Digitalisasi Citra

**📄 File:** `digitalisasi_citra.py`

**🎯 Tujuan:**  
Memahami konsep dasar citra digital: bagaimana citra analog dikonversi ke representasi numerik (matriks piksel), serta menganalisis parameter teknis citra.

**📋 Topik yang Dibahas:**
- Akuisisi citra menggunakan kamera/smartphone
- Representasi matriks (array 2D/3D piksel) dan vektor (flattened array)
- Analisis resolusi spasial, bit depth, aspect ratio, dan ukuran memori
- Manipulasi dasar: cropping, resizing, rotasi

**🔑 Konsep Utama:**

```
Citra RGB (H × W × 3) → Array NumPy 3D
Vektor = img.flatten()  →  Array 1D (H × W × 3 elemen)

Ukuran memori = Total Piksel × Bit Depth / 8 (byte)
```

**💡 Cuplikan Kode Kunci:**
```python
import cv2
import numpy as np

img = cv2.imread("pantai.jpg")
height, width, channels = img.shape

# Representasi matriks (5x5 piksel pertama)
print(img[:5, :5])

# Representasi vektor
vector = img.flatten()

# Manipulasi dasar
crop   = img[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)]
resize = cv2.resize(img, (width//2, height//2))
rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
```

**📊 Parameter yang Dihitung:**

| Parameter | Formula |
|-----------|---------|
| Total piksel | `width × height` |
| Bit depth | `8 × jumlah channel` |
| Ukuran memori | `total_piksel × bit_depth / 8` bytes |
| Aspect ratio | `width / height` |

**🏭 Aplikasi Industri:** Fotografi digital, scanner dokumen, sistem pengawasan CCTV.

---

### Pertemuan 2 — Konversi Ruang Warna & Efek Kuantisasi

**📄 File:** `Analisis_Konversi_Ruang_Warna_dan_Efek_Kuantisasi_untuk_Deteksi_Objek.PY`

**🎯 Tujuan:**  
Menganalisis efek konversi model warna (RGB, Grayscale, HSV, LAB) dan teknik kuantisasi terhadap kemudahan deteksi objek.

**📋 Topik yang Dibahas:**
- Konversi RGB → Grayscale, HSV, LAB menggunakan OpenCV
- Kuantisasi uniform (256 → 16 level intensitas)
- Kuantisasi non-uniform (histogram-based clustering)
- Analisis histogram sebelum/sesudah kuantisasi
- Perbandingan ukuran memori dan waktu komputasi

**🔑 Konsep Utama:**

```
RGB  → Grayscale  : Mengurangi dimensi, cepat
RGB  → HSV        : Memisahkan warna (H) dari kecerahan (V), robust terhadap pencahayaan
RGB  → LAB        : Pemisahan luminansi (L) dan warna (a,b), mendekati persepsi manusia

Kuantisasi Uniform  : intensitas_baru = (intensitas // step) * step
Rasio Kompresi      : ukuran_asli / ukuran_terkuantisasi
```

**💡 Cuplikan Kode Kunci:**
```python
# Konversi ruang warna
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Kuantisasi uniform ke 16 level
levels = 16
step = 256 // levels
quantized = (img // step) * step
```

**📈 Metrik Evaluasi:** Distribusi histogram, ukuran memori (KB/MB), waktu komputasi konversi, kemudahan segmentasi objek.

**🏭 Aplikasi:** Deteksi objek berwarna, sistem pengenalan wajah, inspeksi produk industri.

---

### Pertemuan 3 — Pipeline Transformasi Geometrik & Registrasi Citra

**📄 File:** `Pipeline_Transformasi_Geometrik_untuk_Aplikasi_Registrasi_Citra.py`

**🎯 Tujuan:**  
Mengimplementasikan berbagai transformasi geometrik menggunakan matriks homogen untuk aplikasi registrasi citra (menyelaraskan dua citra objek yang sama dari perspektif berbeda).

**📋 Topik yang Dibahas:**
- Translasi, rotasi, scaling dengan matriks homogen 3×3
- Transformasi affine (estimasi 3 titik korespondensi)
- Transformasi perspektif (estimasi 4 titik)
- Tiga metode interpolasi: Nearest Neighbor, Bilinear, Bicubic
- Evaluasi dengan MSE dan PSNR

**🔑 Konsep Utama:**

```
Matriks Homogen:
  Translasi : T = [[1,0,tx],[0,1,ty],[0,0,1]]
  Rotasi    : R = [[cos θ, -sin θ, 0],[sin θ, cos θ, 0],[0,0,1]]
  Scaling   : S = [[sx,0,0],[0,sy,0],[0,0,1]]

Affine     : 6 DOF (3 titik)  → parallel lines preserved
Perspektif : 8 DOF (4 titik)  → straight lines preserved
```

**💡 Cuplikan Kode Kunci:**
```python
import cv2, numpy as np, math

# Matriks rotasi homogen
angle = 20
theta = np.radians(angle)
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0, 0, 1]], dtype=np.float32)
img_rot = cv2.warpPerspective(img_src, R, (w, h))

# Evaluasi PSNR
def psnr(imgA, imgB):
    mse_val = np.mean((imgA.astype("float") - imgB.astype("float")) ** 2)
    return 20 * math.log10(255.0 / math.sqrt(mse_val))

# Tiga metode interpolasi pada perspektif
img_near   = cv2.warpPerspective(img_src, M, (w,h), interpolation=cv2.INTER_NEAREST)
img_linear = cv2.warpPerspective(img_src, M, (w,h), interpolation=cv2.INTER_LINEAR)
img_cubic  = cv2.warpPerspective(img_src, M, (w,h), interpolation=cv2.INTER_CUBIC)
```

**📊 Perbandingan Interpolasi:**

| Metode | Kualitas | Kecepatan | Cocok Untuk |
|--------|---------|-----------|------------|
| Nearest Neighbor | Rendah | Tercepat | Preview cepat |
| Bilinear | Sedang | Sedang | Keseimbangan kualitas/kecepatan |
| Bicubic | Terbaik | Paling lambat | Output akhir berkualitas tinggi |

---

### Pertemuan 4 — Enhancement Citra Underexposed & Overexposed

**📄 File:** `Pipeline_Enhancement_Citra_Untuk_Optimalisasi_Visual_Citra_Underexposed_dan_Overexposed.py`

**🎯 Tujuan:**  
Meningkatkan kualitas visual citra yang terlalu gelap (underexposed), terlalu terang (overexposed), atau memiliki iluminasi tidak merata menggunakan teknik point processing dan berbasis histogram.

**📋 Topik yang Dibahas:**
- Point processing: negative, log transform, gamma correction (γ = 0.5, 1.5, 2.5)
- Contrast stretching: manual (batas tetap) dan otomatis (percentile-based)
- Histogram Equalization global
- CLAHE (Contrast Limited Adaptive Histogram Equalization) lokal
- Evaluasi: contrast ratio, entropy Shannon

**🔑 Konsep Utama:**

```
Negative      : s = 255 - r
Log Transform : s = c × log(1 + r),  c = 255/log(1 + max)
Gamma (Power) : s = (r/255)^γ × 255

  γ < 1  → mencerahkan (underexposed)
  γ > 1  → menggelapkan (overexposed)

CLAHE vs HE   : HE global, CLAHE lokal per tile → lebih natural
```

**💡 Cuplikan Kode Kunci:**
```python
from skimage.measure import shannon_entropy

# Gamma correction
def gamma_transform(img, gamma):
    norm = img / 255.0
    return np.uint8(np.power(norm, gamma) * 255)

# CLAHE
def clahe(img):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return c.apply(img)

# Metrik
def contrast_ratio(img):  return img.max() - img.min()
def entropy_value(img):   return shannon_entropy(img)
```

**📊 Kapan Menggunakan Teknik Mana:**

| Kondisi Citra | Teknik Terbaik |
|--------------|---------------|
| Underexposed (terlalu gelap) | Log Transform, Gamma γ < 1 |
| Overexposed (terlalu terang) | Gamma γ > 1 |
| Iluminasi tidak merata | CLAHE |
| Histogram sempit/terkonsentrasi | Histogram Equalization |

---

### Pertemuan 5 — Evaluasi Spatial Filtering untuk Restorasi Noise

**📄 File:** `Evaluasi_Spatial_Filtering_untuk_Restorasi_Citra_Terkorupsi_Noise.py`

**🎯 Tujuan:**  
Membandingkan performa berbagai filter spasial (linear dan non-linear) dalam merestorasi citra yang terkorupsi oleh tiga jenis noise yang berbeda.

**📋 Topik yang Dibahas:**
- Pembuatan noise: Gaussian, Salt & Pepper, Speckle
- Filter linear: Mean filter (3×3, 7×7), Gaussian filter (σ=1, σ=2)
- Filter non-linear: Median filter (3×3, 7×7), Min filter
- Evaluasi: MSE, PSNR, SSIM, waktu komputasi

**🔑 Konsep Utama:**

```
Konvolusi Spasial: g(x,y) = f(x,y) ★ h(x,y)

Gaussian Noise   → Gaussian/Mean filter terbaik
Salt & Pepper    → Median filter terbaik (non-linear, mempertahankan tepi)
Speckle Noise    → Gaussian filter efektif

PSNR = 20 × log10(255 / √MSE)  [dB, semakin tinggi = semakin baik]
SSIM ∈ [0,1], 1 = identik
```

**💡 Cuplikan Kode Kunci:**
```python
from skimage.metrics import structural_similarity as ssim

# Definisi semua filter dalam dictionary
filters = {
    "Mean_3x3":       lambda img: cv2.blur(img, (3,3)),
    "Mean_7x7":       lambda img: cv2.blur(img, (7,7)),
    "Gaussian_sigma1": lambda img: cv2.GaussianBlur(img, (5,5), 1),
    "Median_3x3":     lambda img: cv2.medianBlur(img, 3),
    "Min_Filter":     lambda img: cv2.erode(img, np.ones((3,3), np.uint8))
}

# Evaluasi otomatis semua kombinasi noise × filter
for noise_name, noisy_img in noise_images.items():
    for filter_name, filter_func in filters.items():
        restored = filter_func(noisy_img)
        psnr_val = psnr(image, restored)
        ssim_val = ssim(image, restored)
```

**📊 Ringkasan Rekomendasi Filter:**

| Jenis Noise | Filter Terbaik | Alasan |
|-------------|---------------|--------|
| Gaussian | Gaussian σ=1 | Sesuai karakteristik noise |
| Salt & Pepper | Median 3×3 | Non-linear, tidak blur tepi |
| Speckle | Gaussian σ=2 | Smoothing efektif |

---

### Pertemuan 6 — Pipeline Restorasi Motion Blur & Noise

**📄 File:** `Pipeline_Restorasi_Citra_untuk_Motion_Blur_dan_Noise.py`

**🎯 Tujuan:**  
Merestorasi citra yang terdegradasi oleh motion blur menggunakan teknik dekonvolusi berbasis domain frekuensi.

**📋 Topik yang Dibahas:**
- Pemodelan PSF (Point Spread Function) untuk motion blur (arah 30°, panjang 15px)
- Tiga degradasi: blur saja, Gaussian+blur, Salt&Pepper+blur
- Metode restorasi: Inverse Filter, Wiener Filter, Lucy-Richardson Deconvolution
- Visualisasi spektrum frekuensi sebelum/sesudah restorasi

**🔑 Konsep Utama:**

```
Model Degradasi: g(x,y) = f(x,y) ★ h(x,y) + n(x,y)
  g = citra terdegradasi, f = citra asli, h = PSF, n = noise

Inverse Filter  : F̂(u,v) = G(u,v) / H(u,v)          → rawan amplifikasi noise
Wiener Filter   : F̂(u,v) = [H*(u,v) / (|H|² + K)] × G(u,v)  → optimal dengan noise
Lucy-Richardson : iteratif, berbasis statistik Poisson
```

**💡 Cuplikan Kode Kunci:**
```python
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy

# PSF untuk motion blur 15px, sudut 30°
def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1
    return psf / psf.sum()

# Wiener filter
def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)
    F_hat = (np.conj(H) / (np.abs(H)**2 + K)) * G
    return np.abs(np.fft.ifft2(F_hat))
```

**📊 Perbandingan Metode Restorasi:**

| Metode | Kekuatan | Kelemahan |
|--------|---------|----------|
| Inverse Filter | Sederhana | Sangat sensitif noise |
| Wiener Filter | Optimal untuk noise stasioner | Perlu estimasi K yang tepat |
| Lucy-Richardson | Robust, iteratif | Paling lambat |

---

### Pertemuan 7 — Analisis Domain Frekuensi: FFT & Wavelet

**📄 File:** `Analisis_dan_Filtering_Domain_Frekuensi_dengan_FFT_dan_Wavelet.py`

**🎯 Tujuan:**  
Menganalisis dan memfilter citra dalam domain frekuensi menggunakan FFT dan Transformasi Wavelet.

**📋 Topik yang Dibahas:**
- FFT 2D: visualisasi magnitude dan phase spectrum
- Rekonstruksi dari phase saja vs magnitude saja
- Filter frekuensi: Ideal & Gaussian Lowpass/Highpass
- Notch filter untuk menghilangkan noise periodik
- Wavelet db4: dekomposisi 2-level, visualisasi koefisien (cA, cH, cV, cD)
- Perbandingan filtering spasial vs domain frekuensi

**🔑 Konsep Utama:**

```
FFT 2D: F(u,v) = Σ Σ f(x,y) × e^(-j2π(ux/M + vy/N))
Magnitude = log(|F(u,v)| + 1)  → konsentrasi frekuensi
Phase     = ∠F(u,v)            → informasi struktur/tepi

Lowpass  → mempertahankan komponen rendah → smoothing
Highpass → mempertahankan komponen tinggi → deteksi tepi

Wavelet: cA = aproksimasi (LL), cH = horizontal (LH),
         cV = vertikal (HL),    cD = diagonal (HH)
```

**💡 Cuplikan Kode Kunci:**
```python
import pywt

# Gaussian lowpass filter
def gaussian_lowpass(shape, cutoff):
    mask = np.zeros(shape)
    c = shape[0] // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            d = (i-c)**2 + (j-c)**2
            mask[i,j] = np.exp(-d / (2 * cutoff**2))
    return mask

def apply_filter(img, mask):
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.abs(np.fft.ifft2(np.fft.ifftshift(F * mask)))

# Wavelet dekomposisi 2-level
coeffs = pywt.wavedec2(img1, 'db4', level=2)
cA, (cH, cV, cD), (cH2, cV2, cD2) = coeffs
```

**📊 Perbandingan Domain Spasial vs Frekuensi:**

| Aspek | Domain Spasial | Domain Frekuensi |
|-------|---------------|-----------------|
| Implementasi | Sederhana (konvolusi) | Lebih kompleks (FFT) |
| Kontrol frekuensi | Tidak langsung | Presisi tinggi |
| Noise periodik | Tidak efektif | Sangat efektif (notch) |
| Kecepatan (citra besar) | O(N²k²) | O(N² log N) |

---

### ⭐ Pertemuan 8 — UJIAN TENGAH SEMESTER

---

### Pertemuan 9 — Komparasi Metode Segmentasi untuk Ekstraksi Objek

**📄 File:** `Komparasi_Metode_Segmentasi_untuk_Ekstraksi_Objek.py`

**🎯 Tujuan:**  
Membandingkan berbagai metode segmentasi citra untuk memisahkan objek dari latar belakang pada kondisi citra yang bervariasi.

**📋 Topik yang Dibahas:**
- Thresholding: Global (manual T), Otsu's method, Adaptive (Mean & Gaussian)
- Edge detection: Sobel, Prewitt, Canny
- Region-based: Region Growing, Watershed, Connected Components
- Evaluasi: IoU, Dice Coefficient, Accuracy, Precision, Recall
- Overlay kontur hasil segmentasi ke citra asli

**🔑 Konsep Utama:**

```
Ground Truth vs Prediksi:
  TP = piksel objek yang benar terdeteksi
  FP = piksel background salah terdeteksi sebagai objek
  FN = piksel objek yang terlewat

IoU   = TP / (TP + FP + FN)
Dice  = 2×TP / (2×TP + FP + FN)
Acc   = (TP+TN) / Total

Otsu  → optimal untuk histogram bimodal
CLAHE → adaptif untuk iluminasi tidak merata
Watershed → efektif untuk objek overlapping
```

**💡 Cuplikan Kode Kunci:**
```python
# Semua metode dalam pipeline terintegrasi
def thresholding(img):
    res = {}
    _, res["Global"]           = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, res["Otsu"]             = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res["Adaptive Mean"]       = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    res["Adaptive Gaussian"]   = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return res

# Metrik evaluasi
def metrics(gt, pred):
    TP = np.sum((gt==1)&(pred==1))
    FP = np.sum((gt==0)&(pred==1))
    FN = np.sum((gt==1)&(pred==0))
    iou  = TP / (TP + FP + FN + 1e-6)
    dice = 2*TP / (2*TP + FP + FN + 1e-6)
    return iou, dice, ...
```

**📊 Rekomendasi Metode per Jenis Citra:**

| Jenis Citra | Metode Terbaik |
|-------------|---------------|
| Bimodal (kontras tinggi) | Otsu Thresholding |
| Iluminasi tidak merata | Adaptive Gaussian |
| Objek overlapping | Watershed |
| Kebutuhan presisi tepi | Canny Edge Detection |

---

### Pertemuan 10 — Pipeline Morfologi: OCR & Counting Objek

**📄 File:** `Pipeline_Morfologi_untuk_Preprocessing_OCR_dan_Counting_Objek.py`

**🎯 Tujuan:**  
Menerapkan operasi morfologi untuk dua aplikasi praktis: preprocessing dokumen teks sebelum OCR, dan penghitungan objek yang bersentuhan menggunakan kombinasi watershed dan morfologi.

**📋 Topik yang Dibahas:**
- Structuring element: ukuran (3×3, 5×5, 7×7) dan bentuk (square, cross, ellipse)
- Operasi dasar: Erosi, Dilasi (multi-iterasi)
- Operasi majemuk: Opening, Closing, Gradient, Top-hat, Black-hat
- OCR preprocessing pipeline
- Object counting dengan Watershed + Morfologi

**🔑 Konsep Utama:**

```
Erosi   : mengecilkan objek, menghilangkan noise kecil
Dilasi  : membesarkan objek, menutup lubang kecil

Opening = Erosi → Dilasi  (menghilangkan noise, mempertahankan bentuk)
Closing = Dilasi → Erosi  (mengisi hole, menyambung objek terputus)

Gradient  = Dilasi - Erosi  → deteksi boundary
Top-hat   = Original - Opening  → ekstraksi detail terang
Black-hat = Closing - Original  → ekstraksi detail gelap
```

**💡 Cuplikan Kode Kunci:**
```python
# Structuring element dengan variasi bentuk
def get_kernel(size, shape):
    mapping = {"square": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS, "ellipse": cv2.MORPH_ELLIPSE}
    return cv2.getStructuringElement(mapping[shape], (size, size))

# Pipeline OCR preprocessing
kernel = np.ones((3,3), np.uint8)
denoise = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)   # hapus noise
cleaned = cv2.morphologyEx(denoise, cv2.MORPH_CLOSE, kernel)  # sambungkan teks

# Counting dengan Watershed
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
```

**📊 Operasi Morfologi dan Fungsinya:**

| Operasi | Efek | Aplikasi Utama |
|---------|------|---------------|
| Erosi | Mengecilkan/menipis | Hapus noise kecil |
| Dilasi | Membesarkan/menebalkan | Hubungkan komponen terputus |
| Opening | Hapus noise + jaga bentuk | OCR preprocessing |
| Closing | Tutup lubang + sambungkan | Dokumen dengan teks putus |
| Gradient | Highlight boundary | Deteksi tepi morfologi |
| Top-hat | Ekstrak objek terang kecil | Deteksi teks pada background kompleks |

---

### Pertemuan 11 — Shape Analysis Pipeline: Klasifikasi Objek

**📄 File:** `shape_analysis_pipeline.py`

**🎯 Tujuan:**  
Mengekstrak dan menganalisis fitur bentuk objek (moments, chain codes, Fourier descriptors) untuk keperluan klasifikasi menggunakan k-NN.

**📋 Topik yang Dibahas:**
- Properti region: luas, perimeter, centroid, bounding box, convex hull, solidity
- Hu Moments (7 invariant moments): rotasi, skala, dan translasi invariant
- Chain codes: representasi 4-directional dan 8-directional pada kontur
- Fourier descriptors: rekonstruksi batas dengan variasi jumlah komponen (5, 10, 20)
- Klasifikasi k-NN berdasarkan kombinasi fitur terbaik

**🔑 Konsep Utama:**

```
Hu Moments: M = [m1, m2, ..., m7]
  → Invariant terhadap rotasi, translasi, dan skala

Chain Code: representasi arah gerakan 8-tetangga
  0:E, 1:NE, 2:N, 3:NW, 4:W, 5:SW, 6:S, 7:SE

Fourier Descriptors:
  Batas objek → FFT → pilih N koefisien pertama → rekonstruksi
  Semakin banyak N → semakin akurat → lebih mahal komputasi

Classifier k-NN:
  class = majority vote dari k tetangga terdekat di feature space
```

**💡 Cuplikan Kode Kunci:**
```python
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Ekstraksi Hu Moments
moments = cv2.moments(contour)
hu_moments = cv2.HuMoments(moments).flatten()
# Log-transform untuk menstabilkan skala
hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

# Fourier descriptors
contour_complex = contour[:,0,0] + 1j*contour[:,0,1]
fourier_result  = np.fft.fft(contour_complex)
descriptors     = np.abs(fourier_result[:20])  # ambil 20 komponen

# k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
```

---

### Pertemuan 12 — Sistem Pencocokan Objek Berbasis Fitur Lokal

**📄 File:** `Sistem_Pencocokan_Objek_Berbasis_Fitur_Lokal.py`

**🎯 Tujuan:**  
Mengimplementasikan sistem pencocokan objek (object matching) menggunakan fitur lokal yang robust terhadap perubahan rotasi, skala, iluminasi, dan oklusi parsial.

**📋 Topik yang Dibahas:**
- Deteksi dan deskripsi fitur: SIFT, SURF, ORB
- Feature matching: Brute-Force, FLANN, Lowe's ratio test, RANSAC
- Bag of Visual Words (BoVW) dengan k-means clustering (k = 10, 20, 50, 100)
- PCA untuk reduksi dimensi descriptor (16, 32, 64, 128 komponen)
- Evaluasi: kecepatan, akurasi, robustness

**🔑 Konsep Utama:**

```
SIFT : Scale-space extrema → L2 distance → 128-dim descriptor
       Robust terhadap skala, rotasi, pencahayaan
ORB  : Faster (FAST detector + BRIEF descriptor) → Hamming distance
       Real-time, bebas paten

Lowe's Ratio Test: match valid jika dist1/dist2 < 0.75
RANSAC           : estimasi homography, filter outlier geometrik

BoVW Pipeline:
  Ekstraksi fitur → K-means clustering → visual vocabulary
  → histogram visual words per citra → SVM/k-NN classifier
```

**💡 Cuplikan Kode Kunci:**
```python
# SIFT detection & matching
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_ref, None)
kp2, des2 = sift.detectAndCompute(img_test, None)

# FLANN matcher dengan ratio test
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

# RANSAC homography estimation
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

---

### Pertemuan 13 — Komparasi Klasifikasi KNN vs SVM

**📄 File:** `Komparasi_Klasifikasi_KNNvsSVM.py`

**🎯 Tujuan:**  
Melakukan studi komparasi komprehensif antara KNN dan SVM untuk pengenalan citra Fashion-MNIST, termasuk hyperparameter tuning dan analisis trade-off.

**📋 Topik yang Dibahas:**
- Dataset: Fashion-MNIST (10 kelas pakaian, 70.000 gambar)
- Ekstraksi fitur: HOG, histogram warna RGB/HSV, Hu moments, LBP/GLCM tekstur
- KNN: variasi k (1, 3, 5, 7, 9, 11), metrik jarak (Euclidean, Manhattan, Minkowski)
- SVM: kernel (linear, polynomial, RBF), variasi C dan gamma
- Stratified k-fold cross-validation, GridSearchCV, learning curve

**🔑 Konsep Utama:**

```
HOG: Histogram orientasi gradient dalam cell lokal → deskriptor tekstur+bentuk

KNN: class = majority(k nearest neighbors)
  k kecil → overfitting, k besar → underfitting

SVM: hyperplane optimal dengan margin maksimum
  Kernel trick: φ(x) → ruang fitur tinggi dimensi
  C   : trade-off margin vs misclassification
  γ   : bandwidth RBF kernel

Evaluasi:
  Precision = TP/(TP+FP), Recall = TP/(TP+FN)
  F1-score  = 2 × (P×R)/(P+R)
```

**💡 Cuplikan Kode Kunci:**
```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# HOG feature extraction
from skimage.feature import hog
features, hog_image = hog(image, orientations=8, pixels_per_cell=(4,4),
                          cells_per_block=(2,2), visualize=True)

# GridSearchCV untuk SVM
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), param_grid, cv=StratifiedKFold(5), scoring='accuracy')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

**📊 Perbandingan KNN vs SVM:**

| Aspek | KNN | SVM |
|-------|-----|-----|
| Training | Tidak ada (lazy learner) | Lambat (kernel trick) |
| Inference | Lambat (hitung semua jarak) | Cepat |
| Akurasi (fashion) | Sedang | Tinggi |
| Interpretability | Mudah dipahami | Sulit (kernel RBF) |
| Cocok untuk | Dataset kecil | Dataset besar, kompleks |

---

### Pertemuan 14 — Klasifikasi CNN: From Scratch hingga Transfer Learning

**📄 File:** `Klasifikasi_Citra_dengan_CNN_Dari_Awal_hingga_Transfer_Learning.py`

**🎯 Tujuan:**  
Mengimplementasikan dan membandingkan klasifikasi citra menggunakan CNN dari awal, transfer learning (feature extraction), dan fine-tuning pada dataset CIFAR-10.

**📋 Topik yang Dibahas:**
- Arsitektur CNN dari awal: Conv2D → MaxPool → Dropout → Dense
- Transfer learning: VGG16, ResNet50, MobileNetV2 (frozen base layers)
- Fine-tuning: unfreeze beberapa layer terakhir
- Data augmentation: rotasi, flip, zoom, shear, width/height shift
- Visualisasi: feature maps, filter, Grad-CAM, t-SNE embedding

**🔑 Konsep Utama:**

```
CNN Pipeline:
  Input → [Conv → ReLU → Pool]×N → Flatten → Dense → Softmax

Konvolusi: F[i,j] = Σ Σ I[i+m, j+n] × K[m,n]
  → mendeteksi fitur lokal (tepi, tekstur, pola)

Transfer Learning:
  Feature Extraction: freeze semua layer base → train classifier saja
  Fine-tuning        : unfreeze N layer terakhir → train bersama

Data Augmentation → mengurangi overfitting
Grad-CAM → visualisasi area yang diperhatikan model
```

**💡 Cuplikan Kode Kunci:**
```python
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNN dari awal
model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Transfer learning dengan MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False  # Feature extraction
model = Sequential([base_model, layers.GlobalAveragePooling2D(), layers.Dense(10, activation='softmax')])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                              horizontal_flip=True, zoom_range=0.2)
```

**📊 Perbandingan Strategi Training:**

| Strategi | Akurasi | Waktu Training | Data yang Dibutuhkan |
|----------|---------|---------------|---------------------|
| CNN Scratch | Sedang | Lama | Banyak |
| Feature Extraction | Baik | Cepat | Sedikit |
| Fine-tuning | Terbaik | Sedang | Sedang |

---

## 🛠️ Teknologi & Library

```python
# Core
import cv2          # OpenCV - pengolahan citra
import numpy as np  # NumPy - komputasi numerik

# Visualisasi
import matplotlib.pyplot as plt

# Metrik & Analisis Citra
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from skimage.feature import hog

# Pemrosesan Sinyal
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy
import pywt  # PyWavelets

# Machine Learning Klasik
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Deep Learning
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
```

---

## ⚙️ Cara Menjalankan

**1. Clone repositori:**
```bash
git clone https://github.com/DUTAFANROZA15/PENGOLAHAN-CITRA-DIGITAL.git
cd PENGOLAHAN-CITRA-DIGITAL
```

**2. Install dependensi:**
```bash
pip install opencv-python numpy matplotlib scikit-image scipy PyWavelets scikit-learn tensorflow
```

**3. Jalankan program sesuai pertemuan:**
```bash
# Contoh: Pertemuan 5 - Spatial Filtering
python Evaluasi_Spatial_Filtering_untuk_Restorasi_Citra_Terkorupsi_Noise.py
```

> **Catatan:** Pastikan file citra yang dibutuhkan (seperti `kucing.jpg`, `citra.jpeg`, dll.) tersedia di direktori yang sama dengan script, atau sesuaikan `image_path` di dalam kode.

---

## 📊 Ringkasan Metrik & Evaluasi

Berikut adalah metrik evaluasi yang digunakan secara konsisten di seluruh tugas:

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **MSE** | `mean((original - restored)²)` | Semakin kecil = semakin baik |
| **PSNR** | `20 × log10(255 / √MSE)` dB | Semakin tinggi = semakin baik |
| **SSIM** | Perbandingan luminansi, kontras, struktur | 1.0 = identik |
| **IoU** | `TP / (TP+FP+FN)` | 1.0 = segmentasi sempurna |
| **Dice** | `2TP / (2TP+FP+FN)` | Mirip IoU, lebih sensitif TP |
| **F1-Score** | `2 × (P×R) / (P+R)` | Harmonic mean Precision & Recall |

---

## 📖 Referensi

- Gonzalez, R.C. & Woods, R.E. *Digital Image Processing*, 4th Ed. Pearson, 2018.
- Bradski, G. & Kaehler, A. *Learning OpenCV 3*. O'Reilly Media, 2016.
- OpenCV Documentation: https://docs.opencv.org
- Scikit-image Documentation: https://scikit-image.org/docs
- TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs

---

<div align="center">

**Universitas Negeri Padang — Pendidikan Teknik Informatika**  
Mata Kuliah Pengolahan Citra Digital · 2025

</div>
