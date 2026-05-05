import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import time

def praktikum_9_1():
    """
    Perbandingan teknik thresholding: Global, Otsu, dan Adaptive
    """
    print("PRAKTIKUM 9.1: PERBANDINGAN TEKNIK THRESHOLDING")
    print("=" * 60)
    
    # Buat citra test dengan berbagai karakteristik
    def create_test_images():
        images = {}
        
        # 1. Citra bimodal (ideal untuk thresholding)
        img_bimodal = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_bimodal, (30, 30), (150, 150), 50, -1)  # Dark object
        cv2.rectangle(img_bimodal, (100, 100), (220, 220), 200, -1)  # Bright object
        images['Bimodal Image'] = img_bimodal
        
        # 2. Citra dengan uneven illumination
        img_uneven = np.zeros((256, 256), dtype=np.uint8)
        # Create gradient background
        for i in range(256):
            img_uneven[:, i] = i // 2
        # Add objects
        cv2.rectangle(img_uneven, (50, 50), (100, 100), 255, -1)
        cv2.rectangle(img_uneven, (150, 150), (200, 200), 100, -1)
        images['Uneven Illumination'] = img_uneven
        
        # 3. Citra dengan noise
        img_noisy = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_noisy, (50, 50), (150, 150), 128, -1)
        # Add Gaussian noise
        noise = np.random.normal(0, 30, img_noisy.shape)
        img_noisy = np.clip(img_noisy.astype(float) + noise, 0, 255).astype(np.uint8)
        images['Noisy Image'] = img_noisy
        
        # 4. Citra dengan multiple intensity levels
        img_multi = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_multi, (30, 30), (90, 90), 80, -1)   # Dark gray
        cv2.rectangle(img_multi, (100, 30), (160, 90), 120, -1)  # Medium gray
        cv2.rectangle(img_multi, (170, 30), (230, 90), 180, -1)  # Light gray
        images['Multi-level Image'] = img_multi
        
        return images
    
    # Implementasi berbagai metode thresholding
    def apply_global_threshold(image, T=127):
        """Global thresholding"""
        _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary
    
    def apply_otsu_threshold(image):
        """Otsu's thresholding"""
        T_otsu, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, T_otsu
    
    def apply_adaptive_threshold(image, block_size=11, C=2):
        """Adaptive thresholding"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Block size harus ganjil
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, C)
        return binary
    
    def apply_iterative_threshold(image, max_iter=100, tolerance=1):
        """Iterative threshold selection"""
        # Initialize threshold
        T = np.mean(image)
        
        for i in range(max_iter):
            # Segment image
            foreground = image[image > T]
            background = image[image <= T]
            
            # Compute means
            if len(foreground) > 0 and len(background) > 0:
                mu_fg = np.mean(foreground)
                mu_bg = np.mean(background)
                
                # New threshold
                T_new = (mu_fg + mu_bg) / 2
                
                # Check convergence
                if abs(T_new - T) < tolerance:
                    T = T_new
                    break
                    
                T = T_new
            else:
                break
        
        # Apply threshold
        _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary, T
    
    # Buat citra test
    test_images = create_test_images()
    
    # Terapkan berbagai metode thresholding
    results = {}
    
    for name, image in test_images.items():
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply different thresholding methods
        global_binary = apply_global_threshold(gray, 127)
        otsu_binary, T_otsu = apply_otsu_threshold(gray)
        adaptive_binary = apply_adaptive_threshold(gray, 11, 2)
        iterative_binary, T_iter = apply_iterative_threshold(gray)
        
        results[name] = {
            'original': gray,
            'global': global_binary,
            'otsu': otsu_binary,
            'adaptive': adaptive_binary,
            'iterative': iterative_binary,
            'T_otsu': T_otsu,
            'T_iter': T_iter
        }
    
    # Visualisasi hasil
    n_images = len(test_images)
    fig, axes = plt.subplots(n_images, 5, figsize=(20, 4*n_images))
    
    # Handle case when only one image
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(results.items()):
        # Column 1: Original image
        axes[idx, 0].imshow(result['original'], cmap='gray')
        axes[idx, 0].set_title(f'{name}\nOriginal')
        axes[idx, 0].axis('off')
        
        # Column 2-5: Thresholding results
        methods = ['global', 'otsu', 'adaptive', 'iterative']
        titles = ['Global (T=127)', 
                 f'Otsu (T={float(result["T_otsu"]):.0f})', 
                 'Adaptive', 
                 f'Iterative (T={float(result["T_iter"]):.0f})']
        
        for col, (method, title) in enumerate(zip(methods, titles), 1):
            axes[idx, col].imshow(result[method], cmap='gray')
            axes[idx, col].set_title(title)
            axes[idx, col].axis('off')
    
    plt.suptitle('Perbandingan Metode Thresholding', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Analisis histogram dan threshold selection
    print("\nANALISIS HISTOGRAM DAN THRESHOLD SELECTION")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(list(results.items())[:4]):
        # Plot histogram
        hist = cv2.calcHist([result['original']], [0], None, [256], [0, 256])
        axes[idx].plot(hist, 'k-', linewidth=2)
        
        # Add threshold lines
        axes[idx].axvline(x=127, color='r', linestyle='--', label='Global (127)', alpha=0.7)
        axes[idx].axvline(x=float(result['T_otsu']), color='g', linestyle='--', 
                         label=f'Otsu ({float(result["T_otsu"]):.0f})', alpha=0.7)
        axes[idx].axvline(x=float(result['T_iter']), color='b', linestyle='--',
                         label=f'Iterative ({float(result["T_iter"]):.0f})', alpha=0.7)
        
        axes[idx].set_title(f'{name}\nHistogram with Thresholds')
        axes[idx].set_xlabel('Intensity')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 255])
    
    plt.suptitle('Analisis Histogram dengan Posisi Threshold', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Evaluasi kuantitatif (simulasi ground truth)
    print("\nEVALUASI KUANTITATIF (DENGAN SIMULASI GROUND TRUTH)")
    print("-" * 70)
    
    # Buat ground truth untuk bimodal image
    gt_bimodal = np.zeros((256, 256), dtype=np.uint8)
    gt_bimodal[30:150, 30:150] = 1  # First object
    gt_bimodal[100:220, 100:220] = 1  # Second object
    
    # Hitung metrics untuk setiap metode
    def calculate_metrics(binary, ground_truth):
        """Calculate segmentation metrics"""
        # Ensure binary images
        binary = (binary > 0).astype(np.uint8)
        ground_truth = (ground_truth > 0).astype(np.uint8)
        
        # True Positive, False Positive, etc.
        tp = np.sum((binary == 1) & (ground_truth == 1))
        fp = np.sum((binary == 1) & (ground_truth == 0))
        fn = np.sum((binary == 0) & (ground_truth == 1))
        tn = np.sum((binary == 0) & (ground_truth == 0))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou
        }
    
    # Evaluasi untuk bimodal image
    bimodal_result = results['Bimodal Image']
    
    print(f"{'Method':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10}")
    print("-" * 70)
    
    methods = ['global', 'otsu', 'adaptive', 'iterative']
    method_names = ['Global', 'Otsu', 'Adaptive', 'Iterative']
    
    for method, method_name in zip(methods, method_names):
        metrics = calculate_metrics(bimodal_result[method], gt_bimodal)
        print(f"{method_name:<15} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} {metrics['iou']:<10.3f}")
    
    # Visual comparison dengan ground truth
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original and ground truth
    axes[0, 0].imshow(bimodal_result['original'], cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_bimodal, cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')  # Empty
    
    # Row 2: Thresholding results dengan overlay errors
    for idx, (method, method_name) in enumerate(zip(methods[:3], method_names[:3])):
        result_binary = (bimodal_result[method] > 0).astype(np.uint8)
        
        # Create error visualization
        error_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # True Positive: White
        tp_mask = (result_binary == 1) & (gt_bimodal == 1)
        error_image[tp_mask] = [255, 255, 255]
        
        # False Positive: Red (segmented but not in GT)
        fp_mask = (result_binary == 1) & (gt_bimodal == 0)
        error_image[fp_mask] = [255, 0, 0]
        
        # False Negative: Blue (in GT but not segmented)
        fn_mask = (result_binary == 0) & (gt_bimodal == 1)
        error_image[fn_mask] = [0, 0, 255]
        
        axes[1, idx].imshow(error_image)
        axes[1, idx].set_title(f'{method_name}\n(Red: FP, Blue: FN)', fontsize=12)
        axes[1, idx].axis('off')
    
    plt.suptitle('Visualisasi Error Segmentation', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Analisis tambahan: Pengaruh parameter pada adaptive thresholding
    print("\nANALISIS PENGARUH PARAMETER ADAPTIVE THRESHOLDING")
    print("-" * 60)
    
    uneven_image = results['Uneven Illumination']['original']
    
    # Buat figure dengan ukuran yang sesuai (2 baris, 4 kolom)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Test different block sizes (baris 1)
    block_sizes = [3, 11, 31, 51]
    for idx, block_size in enumerate(block_sizes):
        adaptive_result = apply_adaptive_threshold(uneven_image, block_size, 2)
        axes[0, idx].imshow(adaptive_result, cmap='gray')
        axes[0, idx].set_title(f'Block Size = {block_size}, C=2', fontsize=10)
        axes[0, idx].axis('off')
    
    # Test different C values (baris 2)
    C_values = [0, 2, 5, 10]
    for idx, C in enumerate(C_values):
        adaptive_result = apply_adaptive_threshold(uneven_image, 11, C)
        axes[1, idx].imshow(adaptive_result, cmap='gray')
        axes[1, idx].set_title(f'Block Size=11, C={C}', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.suptitle('Pengaruh Parameter pada Adaptive Thresholding\n(Baris 1: Variasi Block Size, Baris 2: Variasi C)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Analisis tambahan: Perbandingan waktu komputasi
    print("\nANALISIS WAKTU KOMPUTASI")
    print("-" * 60)
    
    test_image = results['Bimodal Image']['original']
    
    times = {}
    
    # Ukur waktu untuk setiap metode
    start = time.time()
    for _ in range(100):
        apply_global_threshold(test_image, 127)
    times['Global'] = time.time() - start
    
    start = time.time()
    for _ in range(100):
        apply_otsu_threshold(test_image)
    times['Otsu'] = time.time() - start
    
    start = time.time()
    for _ in range(100):
        apply_adaptive_threshold(test_image, 11, 2)
    times['Adaptive'] = time.time() - start
    
    start = time.time()
    for _ in range(100):
        apply_iterative_threshold(test_image)
    times['Iterative'] = time.time() - start
    
    # Plot waktu komputasi
    methods = list(times.keys())
    time_values = list(times.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, time_values, color=['red', 'green', 'blue', 'orange'])
    plt.xlabel('Metode Thresholding')
    plt.ylabel('Waktu Komputasi (detik)')
    plt.title('Perbandingan Waktu Komputasi (100 iterasi)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Tambahkan nilai pada bar
    for bar, value in zip(bars, time_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nWaktu komputasi (100 iterasi):")
    for method, t in times.items():
        print(f"  {method}: {t:.4f} detik")
    
    # Kesimpulan dan rekomendasi (VERSI RINGKAS)
    print("\n" + "="*60)
    print("RINGKASAN THRESHOLDING")
    print("="*60)
    print("""
METODE          | KELEBIHAN              | KEKURANGAN                | PENGGUNAAN
────────────────┼───────────────────────┼───────────────────────────┼─────────────────
Global          | Cepat, sederhana       | Manual, gagal uneven      | Kontras tinggi
Otsu            | Otomatis, optimal      | Gagal multi-modal/uneven  | Bimodal
Adaptive        | Tahan uneven           | Perlu tuning, berat       | Iluminasi tidak merata
Iterative       | Self-tuning            | Berat, bisa divergen      | Non-bimodal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PANDUAN MEMILIH METODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kontras tinggi        → Global (manual)
Bimodal histogram     → Otsu
Uneven illumination   → Adaptive
Banyak noise          → Adaptive + preprocessing
Multi-level           → Multi-level thresholding
Real-time             → Global atau Otsu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER ADAPTIVE THRESHOLDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Block size kecil (3-11)  → Detail bagus, sensitif noise
Block size besar (31-51) → Halus, bisa kehilangan detail
Nilai C kecil            → Sensitif variasi intensitas
Nilai C besar            → Toleran variasi intensitas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIPS IMPLEMENTASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Preprocessing (denoising) sebelum thresholding
2. Eksperimen parameter untuk hasil optimal
3. Kombinasi dengan morphological operations
4. Evaluasi dengan ground truth
5. Trade-off akurasi vs kecepatan
""")
    
    return results

# Jalankan program
if __name__ == "__main__":
    thresholding_results = praktikum_9_1()
    
    # Informasi tambahan
    print("\n" + "="*60)
    print("PROGRAM SELESAI")
    print("="*60)
    print(f"Total images processed: {len(thresholding_results)}")
    print(f"Methods compared: Global, Otsu, Adaptive, Iterative")
    print("\nTeknik yang dipelajari:")
    print("✓ Thresholding global manual")
    print("✓ Otsu's automatic thresholding")
    print("✓ Adaptive thresholding dengan Gaussian window")
    print("✓ Iterative threshold selection")
    print("✓ Evaluasi kuantitatif (Accuracy, Precision, Recall, F1-Score, IoU)")
    print("✓ Visualisasi error segmentation")
    print("✓ Analisis pengaruh parameter adaptive thresholding")
    print("✓ Perbandingan waktu komputasi")