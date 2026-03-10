# ============================================
# LATIHAN 2: MEDICAL IMAGE ENHANCEMENT PIPELINE (VERSI FINAL DIPERBAIKI)
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================
# FUNGSI ENTROPY
# ============================================

def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist_norm = hist.ravel() / hist.sum()

    entropy = 0
    for p in hist_norm:
        if p > 0:
            entropy += -p * np.log2(p)

    return entropy


# ============================================
# MEDICAL IMAGE ENHANCEMENT FUNCTION
# ============================================

def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for medical images
    
    Parameters:
    medical_image : input grayscale medical image
    modality : 'X-ray', 'MRI', 'CT', 'Ultrasound'
    
    Returns:
    enhanced_image
    enhancement report
    """

    # Pastikan grayscale
    if len(medical_image.shape) == 3:
        medical_image = cv2.cvtColor(medical_image, cv2.COLOR_BGR2GRAY)

    enhanced = medical_image.copy()
    steps = []

    # ============================================
    # ENHANCEMENT BERDASARKAN MODALITY
    # ============================================

    if modality == 'X-ray':

        # CLAHE meningkatkan kontras tulang
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
        steps.append("CLAHE Contrast Enhancement")

        # Reduksi noise
        enhanced = cv2.GaussianBlur(enhanced,(3,3),0)
        steps.append("Gaussian Noise Reduction")


    elif modality == 'MRI':

        # Median filter
        enhanced = cv2.medianBlur(enhanced,3)
        steps.append("Median Filtering")

        # Contrast stretching
        p2, p98 = np.percentile(enhanced,(2,98))
        enhanced = exposure.rescale_intensity(enhanced, in_range=(p2,p98))
        enhanced = (enhanced*255).astype(np.uint8)
        steps.append("Contrast Stretching")


    elif modality == 'CT':

        # Edge preserving smoothing
        enhanced = cv2.bilateralFilter(enhanced,9,75,75)
        steps.append("Bilateral Filtering")

        # Histogram Equalization
        enhanced = cv2.equalizeHist(enhanced)
        steps.append("Histogram Equalization")


    elif modality == 'Ultrasound':

        # Speckle noise reduction
        enhanced = cv2.medianBlur(enhanced,5)
        steps.append("Speckle Noise Reduction")

        # Adaptive contrast
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        enhanced = clahe.apply(enhanced)
        steps.append("Adaptive Contrast Enhancement")

    else:
        steps.append("No enhancement applied")


    # ============================================
    # METRICS ANALYSIS
    # ============================================

    psnr_value = psnr(medical_image, enhanced)
    ssim_value = ssim(medical_image, enhanced)

    entropy_original = calculate_entropy(medical_image)
    entropy_enhanced = calculate_entropy(enhanced)

    report = {
        "Modality": modality,
        "Enhancement Steps": steps,
        "PSNR": round(psnr_value,3),
        "SSIM": round(ssim_value,3),
        "Entropy Original": round(entropy_original,3),
        "Entropy Enhanced": round(entropy_enhanced,3)
    }

    return enhanced, report


# ============================================
# MAIN PROGRAM
# ============================================

if __name__ == "__main__":

    print("===================================")
    print("MEDICAL IMAGE ENHANCEMENT SYSTEM")
    print("===================================")

    # Load image
    image = cv2.imread("medical.jpeg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    # Pilih modality
    modality = "X-ray"   # bisa diganti: MRI / CT / Ultrasound

    # Jalankan enhancement
    enhanced_image, report = medical_image_enhancement(image, modality)


    # ============================================
    # VISUALISASI HASIL
    # ============================================

    plt.figure(figsize=(12,8))

    # Original image
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Medical Image")
    plt.axis("off")

    # Enhanced image
    plt.subplot(2,2,2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title("Enhanced Image")
    plt.axis("off")

    # Histogram original
    plt.subplot(2,2,3)
    plt.hist(image.ravel(), bins=256, range=(0,256))
    plt.title("Original Histogram")

    # Histogram enhanced
    plt.subplot(2,2,4)
    plt.hist(enhanced_image.ravel(), bins=256, range=(0,256))
    plt.title("Enhanced Histogram")

    plt.tight_layout()
    plt.show()


    # ============================================
    # SIMPAN HASIL
    # ============================================

    cv2.imwrite("enhanced_medical_image.png", enhanced_image)


    # ============================================
    # CETAK LAPORAN
    # ============================================

    print("\n=== ENHANCEMENT REPORT ===")

    print("Modality :", report["Modality"])

    print("\nEnhancement Steps:")
    for step in report["Enhancement Steps"]:
        print("-", step)

    print("\nImage Quality Metrics")
    print("PSNR :", report["PSNR"])
    print("SSIM :", report["SSIM"])

    print("\nEntropy Analysis")
    print("Entropy Original :", report["Entropy Original"])
    print("Entropy Enhanced :", report["Entropy Enhanced"])

    print("\nHasil citra disimpan sebagai: enhanced_medical_image.png")
