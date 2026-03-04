# ==========================================================
# TRANSFORMASI GEOMETRIK + INTERPOLASI + ANALISIS
# PIPELINE REGISTRASI CITRA
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================================
# 1. LOAD DUA CITRA
# ==========================================================

img_lurus = cv2.imread("lurus.jpg")
img_miring = cv2.imread("miring.jpg")

if img_lurus is None or img_miring is None:
    print("File lurus.jpg atau miring.jpg tidak ditemukan!")
    exit()

# Simpan salinan asli
img_lurus_asli = cv2.cvtColor(img_lurus, cv2.COLOR_BGR2RGB)
img_miring_asli = cv2.cvtColor(img_miring, cv2.COLOR_BGR2RGB)

# Konversi ke RGB untuk tampilan
img_lurus = cv2.cvtColor(img_lurus, cv2.COLOR_BGR2RGB)
img_miring = cv2.cvtColor(img_miring, cv2.COLOR_BGR2RGB)

h, w = img_lurus.shape[:2]

# ==========================================================
# 2. FUNGSI MSE & PSNR
# ==========================================================

def hitung_mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def hitung_psnr(img1, img2):
    mse = hitung_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255 ** 2) / mse)

# ==========================================================
# 3. FUNGSI VISUALISASI DENGAN JUDUL HALAMAN
# ==========================================================

def tampilkan_dengan_judul_halaman(gambar, judul_halaman, keterangan=None):
    """
    Menampilkan gambar dengan judul halaman di bagian atas
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Judul halaman
    fig.suptitle(judul_halaman, fontsize=16, fontweight='bold', y=0.98)
    
    # Tampilkan gambar
    ax = plt.subplot(111)
    ax.imshow(gambar)
    ax.axis("off")
    
    # Keterangan tambahan (opsional)
    if keterangan:
        plt.figtext(0.5, 0.02, keterangan, ha='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Beri ruang untuk judul
    plt.show()

def tampilkan_grid_dengan_judul_halaman(gambar_list, judul_list, judul_halaman, baris, kolom, figsize=(15,10)):
    """
    Menampilkan grid gambar dengan judul halaman di atas
    """
    fig, axes = plt.subplots(baris, kolom, figsize=figsize)
    axes = axes.ravel()
    
    # Judul halaman
    fig.suptitle(judul_halaman, fontsize=16, fontweight='bold', y=0.98)
    
    for i in range(len(gambar_list)):
        axes[i].imshow(gambar_list[i])
        axes[i].axis('off')
        
        # Keterangan singkat di bawah subplot
        axes[i].text(0.5, -0.1, judul_list[i], 
                    transform=axes[i].transAxes,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    # Hide unused subplots
    for i in range(len(gambar_list), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1)  # Beri ruang untuk judul halaman
    plt.show()

def tampilkan_perbandingan_dengan_judul_halaman(gambar1, gambar2, judul_halaman, judul1, judul2):
    """
    Menampilkan perbandingan dengan judul halaman
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Judul halaman
    fig.suptitle(judul_halaman, fontsize=16, fontweight='bold', y=0.98)
    
    # Gambar 1
    axes[0].imshow(gambar1)
    axes[0].axis('off')
    axes[0].text(0.5, -0.1, judul1, transform=axes[0].transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    # Gambar 2
    axes[1].imshow(gambar2)
    axes[1].axis('off')
    axes[1].text(0.5, -0.1, judul2, transform=axes[1].transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    # Difference image
    diff = np.abs(gambar1.astype(float) - gambar2.astype(float))
    diff = diff / diff.max() * 255
    axes[2].imshow(diff.astype(np.uint8), cmap='hot')
    axes[2].axis('off')
    axes[2].text(0.5, -0.1, 'Difference Map', transform=axes[2].transAxes,
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)  # Beri ruang untuk judul
    plt.show()

# ==========================================================
# 4. TRANSLASI
# ==========================================================

print("\n===== MEMPROSES TRANSFORMASI =====\n")

T = np.array([
    [1, 0, 100],
    [0, 1, 50],
    [0, 0, 1]
], dtype=np.float32)

start = time.time()
translasi = cv2.warpPerspective(img_lurus, T, (w, h))
waktu_translasi = time.time() - start
print(f"✓ Translasi selesai dalam {waktu_translasi:.4f} detik")

# ==========================================================
# 5. ROTASI
# ==========================================================

theta = np.radians(30)

R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
], dtype=np.float32)

start = time.time()
rotasi = cv2.warpPerspective(img_lurus, R, (w, h))
waktu_rotasi = time.time() - start
print(f"✓ Rotasi selesai dalam {waktu_rotasi:.4f} detik")

# ==========================================================
# 6. SCALING DENGAN 4 METODE INTERPOLASI
# ==========================================================

S = np.array([
    [0.7, 0, 0],
    [0, 0.7, 0],
    [0, 0, 1]
], dtype=np.float32)

hasil_interpolasi = {}

metode_interpolasi = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4
}

print("\n----- ANALISIS INTERPOLASI SCALING -----")
for nama, flag in metode_interpolasi.items():
    start = time.time()
    img_scaled = cv2.warpPerspective(img_lurus, S, (w, h), flags=flag)
    waktu = time.time() - start

    mse = hitung_mse(img_lurus, img_scaled)
    psnr = hitung_psnr(img_lurus, img_scaled)

    hasil_interpolasi[nama] = {
        "gambar": img_scaled,
        "MSE": mse,
        "PSNR": psnr,
        "Waktu": waktu
    }
    print(f"  {nama:10} | MSE: {mse:.2f} | PSNR: {psnr:.2f} dB | Waktu: {waktu:.6f} detik")

# ==========================================================
# 7. TRANSFORMASI AFFINE
# ==========================================================

print("\n----- TRANSFORMASI AFFINE -----")

pts_src_affine = np.float32([[0,0], [w-1,0], [0,h-1]])
pts_dst_affine = np.float32([[50,50], [w-100,30], [70,h-80]])

start = time.time()
M_affine = cv2.getAffineTransform(pts_src_affine, pts_dst_affine)
affine_result = cv2.warpAffine(img_lurus, M_affine, (w, h))
waktu_affine = time.time() - start

mse_affine = hitung_mse(img_lurus, affine_result)
psnr_affine = hitung_psnr(img_lurus, affine_result)

print(f"  Affine selesai | MSE: {mse_affine:.2f} | PSNR: {psnr_affine:.2f} dB | Waktu: {waktu_affine:.4f} detik")

# ==========================================================
# 8. TRANSFORMASI PERSPEKTIF (REGISTRASI) - DIPERBAIKI
# ==========================================================

print("\n----- TRANSFORMASI PERSPEKTIF (REGISTRASI) -----")

# Titik kontrol untuk transformasi perspektif (SESUAI DATA YANG DIBERIKAN)
pts_src_persp = np.float32([
    [92.0, 332.0],    # P1: Kiri atas (x=92, y=332)
    [568.0, 107.0],   # P2: Kanan atas (x=568, y=107)
    [664.0, 1272.0],  # P3: Kanan bawah (x=664, y=1272)
    [71.0, 1155.0]    # P4: Kiri bawah (x=71, y=1155)
])

# Titik tujuan (membentuk persegi panjang sesuai dimensi citra)
pts_dst_persp = np.float32([
    [0, 0],           # Kiri atas
    [w-1, 0],         # Kanan atas
    [w-1, h-1],       # Kanan bawah
    [0, h-1]          # Kiri bawah
])

# Hitung transformasi
start = time.time()
M_persp = cv2.getPerspectiveTransform(pts_src_persp, pts_dst_persp)
perspektif = cv2.warpPerspective(img_miring, M_persp, (w, h))
waktu_perspektif = time.time() - start

# Evaluasi hasil registrasi
mse_perspektif = hitung_mse(img_lurus, perspektif)
psnr_perspektif = hitung_psnr(img_lurus, perspektif)

print(f"  Perspektif selesai | MSE: {mse_perspektif:.2f} | PSNR: {psnr_perspektif:.2f} dB | Waktu: {waktu_perspektif:.4f} detik")

# Tampilkan koordinat titik kontrol yang digunakan
print("\n----- KOORDINAT TITIK KONTROL YANG DIGUNAKAN -----")
print("Titik\tSumber (x,y)\t\tTujuan (x,y)")
print("-" * 55)
for i in range(4):
    print(f"P{i+1}\t({pts_src_persp[i][0]:.1f}, {pts_src_persp[i][1]:.1f})\t\t({pts_dst_persp[i][0]:.0f}, {pts_dst_persp[i][1]:.0f})")

# ==========================================================
# 9. ANALISIS HASIL
# ==========================================================

print("\n" + "="*60)
print(" ANALISIS HASIL")
print("="*60)

# Analisis interpolasi
terbaik_psnr = max(hasil_interpolasi, key=lambda x: hasil_interpolasi[x]["PSNR"])
tercepat = min(hasil_interpolasi, key=lambda x: hasil_interpolasi[x]["Waktu"])

print("\n--- METODE INTERPOLASI ---")
print(f"  Tertinggi PSNR : {terbaik_psnr} ({hasil_interpolasi[terbaik_psnr]['PSNR']:.2f} dB)")
print(f"  Tercepat       : {tercepat} ({hasil_interpolasi[tercepat]['Waktu']:.6f} detik)")
print(f"  Selisih PSNR   : {hasil_interpolasi[terbaik_psnr]['PSNR'] - hasil_interpolasi[tercepat]['PSNR']:.2f} dB")

# Analisis affine vs perspektif
print("\n--- AFFINE VS PERSPEKTIF ---")
print(f"  Affine     | PSNR: {psnr_affine:.2f} dB | Waktu: {waktu_affine:.4f} s")
print(f"  Perspektif | PSNR: {psnr_perspektif:.2f} dB | Waktu: {waktu_perspektif:.4f} s")
print(f"  Selisih    | PSNR: {psnr_perspektif - psnr_affine:.2f} dB")

if psnr_perspektif > psnr_affine:
    print("\n  ✓ Perspektif lebih baik untuk kasus ini (registrasi dokumen miring)")
else:
    print("\n  ✓ Affine lebih cepat dan cukup untuk transformasi sederhana")

# ==========================================================
# 10. VISUALISASI HASIL (DENGAN JUDUL HALAMAN)
# ==========================================================

print("\n===== MENAMPILKAN VISUALISASI =====\n")

# HALAMAN 1: Citra Asli
gambar_asli = [img_lurus_asli, img_miring_asli]
judul_asli = ["Citra Lurus (Referensi)", "Citra Miring (Akan Diregistrasi)"]
tampilkan_grid_dengan_judul_halaman(
    gambar_asli, 
    judul_asli, 
    "HALAMAN 1: CITRA INPUT", 
    1, 2, 
    (14, 7)
)

# HALAMAN 2: Transformasi Dasar
gambar_dasar = [translasi, rotasi]
judul_dasar = [
    f"Translasi (dx=100, dy=50)\nPSNR: {hitung_psnr(img_lurus, translasi):.2f} dB | {waktu_translasi:.4f}s",
    f"Rotasi 30°\nPSNR: {hitung_psnr(img_lurus, rotasi):.2f} dB | {waktu_rotasi:.4f}s"
]
tampilkan_grid_dengan_judul_halaman(
    gambar_dasar, 
    judul_dasar, 
    "HALAMAN 2: TRANSFORMASI DASAR (TRANSLASI & ROTASI)", 
    1, 2, 
    (14, 7)
)

# HALAMAN 3: Perbandingan Metode Interpolasi
gambar_interp = [hasil_interpolasi[n]["gambar"] for n in hasil_interpolasi]
judul_interp = [
    f"{n}\nPSNR: {hasil_interpolasi[n]['PSNR']:.2f} dB | {hasil_interpolasi[n]['Waktu']:.6f}s"
    for n in hasil_interpolasi
]
tampilkan_grid_dengan_judul_halaman(
    gambar_interp, 
    judul_interp, 
    "HALAMAN 3: PERBANDINGAN METODE INTERPOLASI (SCALING 0.7x)", 
    2, 2, 
    (16, 12)
)

# HALAMAN 4: Transformasi Affine vs Perspektif
gambar_aff_persp = [affine_result, perspektif]
judul_aff_persp = [
    f"Affine (3 titik)\nPSNR: {psnr_affine:.2f} dB | {waktu_affine:.4f}s",
    f"Perspektif - HASIL REGISTRASI\nPSNR: {psnr_perspektif:.2f} dB | {waktu_perspektif:.4f}s"
]
tampilkan_grid_dengan_judul_halaman(
    gambar_aff_persp, 
    judul_aff_persp, 
    "HALAMAN 4: TRANSFORMASI LANJUTAN (AFFINE vs PERSPEKTIF)", 
    1, 2, 
    (14, 7)
)

# HALAMAN 5: Evaluasi Registrasi
print("\n----- HALAMAN 5: EVALUASI REGISTRASI -----")
tampilkan_perbandingan_dengan_judul_halaman(
    img_lurus_asli, 
    perspektif,
    "HALAMAN 5: EVALUASI HASIL REGISTRASI PERSPEKTIF",
    "Citra Referensi (Lurus)", 
    "Hasil Registrasi Perspektif"
)

# HALAMAN 6: Titik Kontrol (DIPERBAIKI DENGAN KOORDINAT BARU)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("HALAMAN 6: TITIK KONTROL TRANSFORMASI PERSPEKTIF", fontsize=16, fontweight='bold', y=0.98)

# Gambar kiri dengan titik kontrol sumber
axes[0].imshow(img_miring_asli)
for i, pt in enumerate(pts_src_persp):
    axes[0].plot(pt[0], pt[1], 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
    axes[0].text(pt[0]+30, pt[1]-20, f'P{i+1}\n({pt[0]:.0f}, {pt[1]:.0f})', 
                color='red', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='red'))
axes[0].set_title("Citra Miring dengan Titik Kontrol Sumber", fontsize=12)
axes[0].axis('off')

# Gambar kanan dengan titik kontrol tujuan
axes[1].imshow(img_lurus_asli)
for i, pt in enumerate(pts_dst_persp):
    axes[1].plot(pt[0], pt[1], 'go', markersize=12, markeredgecolor='white', markeredgewidth=2)
    axes[1].text(pt[0]+30, pt[1]-20, f'P{i+1}\'\n({pt[0]:.0f}, {pt[1]:.0f})', 
                color='green', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='green'))
axes[1].set_title("Citra Referensi dengan Titik Kontrol Tujuan", fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.05)
plt.show()

# Tabel koordinat titik kontrol yang diperbaiki
print("\n" + "="*60)
print(" KOORDINAT TITIK KONTROL (YANG DIPERBAIKI)")
print("="*60)
print("\n{:<10} {:<25} {:<25}".format("Titik", "Sumber (x,y) - Citra Miring", "Tujuan (x,y) - Citra Lurus"))
print("-" * 65)
for i in range(4):
    print(f"P{i+1}\t({pts_src_persp[i][0]:.1f}, {pts_src_persp[i][1]:.1f})\t\t({pts_dst_persp[i][0]:.0f}, {pts_dst_persp[i][1]:.0f})")

# Informasi tambahan tentang titik kontrol
print("\n--- INFORMASI TITIK KONTROL ---")
print(f"P1 (Kiri Atas)    : ({pts_src_persp[0][0]:.1f}, {pts_src_persp[0][1]:.1f}) → (0, 0)")
print(f"P2 (Kanan Atas)   : ({pts_src_persp[1][0]:.1f}, {pts_src_persp[1][1]:.1f}) → ({w-1}, 0)")
print(f"P3 (Kanan Bawah)  : ({pts_src_persp[2][0]:.1f}, {pts_src_persp[2][1]:.1f}) → ({w-1}, {h-1})")
print(f"P4 (Kiri Bawah)   : ({pts_src_persp[3][0]:.1f}, {pts_src_persp[3][1]:.1f}) → (0, {h-1})")

# ==========================================================
# 11. RINGKASAN AKHIR
# ==========================================================

print("\n" + "="*60)
print(" RINGKASAN HASIL EVALUASI")
print("="*60)

print("\n{:<20} {:>12} {:>15} {:>15}".format(
    "Transformasi", "MSE", "PSNR (dB)", "Waktu (s)"))
print("-" * 65)

# Translasi
print("{:<20} {:>12.2f} {:>15.2f} {:>15.4f}".format(
    "Translasi", hitung_mse(img_lurus, translasi), 
    hitung_psnr(img_lurus, translasi), waktu_translasi))

# Rotasi
print("{:<20} {:>12.2f} {:>15.2f} {:>15.4f}".format(
    "Rotasi", hitung_mse(img_lurus, rotasi), 
    hitung_psnr(img_lurus, rotasi), waktu_rotasi))

# Interpolasi
for nama in hasil_interpolasi:
    print("{:<20} {:>12.2f} {:>15.2f} {:>15.6f}".format(
        f"Scaling-{nama}", hasil_interpolasi[nama]["MSE"], 
        hasil_interpolasi[nama]["PSNR"], hasil_interpolasi[nama]["Waktu"]))

# Affine
print("{:<20} {:>12.2f} {:>15.2f} {:>15.4f}".format(
    "Affine", mse_affine, psnr_affine, waktu_affine))

# Perspektif
print("{:<20} {:>12.2f} {:>15.2f} {:>15.4f}".format(
    "Perspektif", mse_perspektif, psnr_perspektif, waktu_perspektif))

print("\n" + "="*60)
print(" KESIMPULAN")
print("="*60)

print(f"""
✓ INTERPOLASI:
   - Terbaik (PSNR): {terbaik_psnr} ({hasil_interpolasi[terbaik_psnr]['PSNR']:.2f} dB)
   - Tercepat: {tercepat} ({hasil_interpolasi[tercepat]['Waktu']:.6f} detik)

✓ REGISTRASI:
   - Metode: Transformasi Perspektif (Homografi) dengan 4 titik kontrol
   - PSNR: {psnr_perspektif:.2f} dB
   - Waktu: {waktu_perspektif:.4f} detik

✓ VERDIK: Transformasi Perspektif berhasil meregistrasi citra miring 
   dengan kualitas {'BAIK' if psnr_perspektif > 20 else 'CUKUP' if psnr_perspektif > 15 else 'PERLU PENINGKATAN'}
""")

print("="*60)