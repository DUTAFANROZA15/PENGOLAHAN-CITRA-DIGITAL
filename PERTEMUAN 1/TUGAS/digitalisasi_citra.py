import cv2
import numpy as np
import os

# ===============================
# 1. MEMBACA CITRA
# ===============================

# Ganti path sesuai lokasi file Anda
image_path = "pantai.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Citra tidak ditemukan!")
    exit()

print("Citra berhasil dibaca.\n")

# ===============================
# 2. REPRESENTASI MATRKS & VEKTOR
# ===============================

height, width, channels = img.shape

print("=== INFORMASI CITRA ===")
print("Resolusi        :", width, "x", height)
print("Jumlah Channel  :", channels)
print("Shape           :", img.shape)
print()

print("=== 5x5 PIKSEL PERTAMA ===")
print(img[:5, :5])
print()

vector = img.flatten()

print("=== 20 ELEMEN PERTAMA VEKTOR ===")
print(vector[:20])
print()

# ===============================
# 3. ANALISIS PARAMETER CITRA
# ===============================

# Resolusi Spasial
total_pixels = width * height

# Bit depth (8 bit per channel)
bit_depth = 8 * channels

# Total warna
total_colors = 2 ** bit_depth

# Aspect ratio
aspect_ratio = width / height

# Ukuran memori
memory_bytes = total_pixels * bit_depth / 8
memory_kb = memory_bytes / 1024
memory_mb = memory_kb / 1024

print("=== ANALISIS PARAMETER ===")
print("Total Piksel        :", total_pixels)
print("Bit Depth           :", bit_depth, "bit")
print("Jumlah Warna        :", total_colors)
print("Aspect Ratio        :", round(aspect_ratio, 2))
print("Ukuran Memori       :", round(memory_mb, 2), "MB")
print()

# ===============================
# SIMULASI PERUBAHAN
# ===============================

new_width = width * 2
new_height = height * 2
new_bit_depth = bit_depth / 2

new_total_pixels = new_width * new_height
new_memory_bytes = new_total_pixels * new_bit_depth / 8
new_memory_mb = (new_memory_bytes / 1024) / 1024

print("=== SIMULASI PERUBAHAN ===")
print("Resolusi Baru       :", new_width, "x", new_height)
print("Bit Depth Baru      :", new_bit_depth, "bit")
print("Ukuran Memori Baru  :", round(new_memory_mb, 2), "MB")
print()

# ===============================
# 4. MANIPULASI DASAR
# ===============================

# 1. Cropping
crop = img[int(height*0.2):int(height*0.8),
           int(width*0.2):int(width*0.8)]

# 2. Resizing (setengah ukuran)
resize = cv2.resize(img, (width//2, height//2))

# 3. Rotasi 90 derajat
rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# ===============================
# MENYIMPAN HASIL
# ===============================

cv2.imwrite("hasil_crop.jpg", crop)
cv2.imwrite("hasil_resize.jpg", resize)
cv2.imwrite("hasil_rotate.jpg", rotate)

print("Manipulasi citra selesai.")
print("File disimpan sebagai:")
print("- hasil_crop.jpg")
print("- hasil_resize.jpg")
print("- hasil_rotate.jpg")
print()

# ===============================
# MENAMPILKAN HASIL
# ===============================

cv2.imshow("Citra Asli", img)
cv2.imshow("Crop", crop)
cv2.imshow("Resize", resize)
cv2.imshow("Rotate", rotate)

cv2.waitKey(0)
cv2.destroyAllWindows()
