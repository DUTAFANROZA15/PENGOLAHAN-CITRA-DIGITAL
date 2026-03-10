# ============================================
# LATIHAN 3: REAL-TIME VIDEO ENHANCEMENT
# ============================================

import cv2
import numpy as np
import time
from collections import deque

class RealTimeEnhancement:
    
    def __init__(self, target_fps=30, buffer_size=5):
        """
        Inisialisasi sistem enhancement
        
        Parameters:
        target_fps : target frame per second
        buffer_size : jumlah frame history untuk temporal consistency
        """
        self.target_fps = target_fps
        self.history_buffer = deque(maxlen=buffer_size)
        self.frame_time = 1.0 / target_fps
        
        # CLAHE untuk adaptive enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

    # ============================================
    # Contrast Stretching
    # ============================================
    def contrast_stretch(self, image):
        
        min_val = np.min(image)
        max_val = np.max(image)

        stretched = (image - min_val) * (255/(max_val - min_val + 1e-5))
        stretched = np.clip(stretched,0,255)

        return stretched.astype(np.uint8)

    # ============================================
    # Gamma Correction
    # ============================================
    def gamma_correction(self, image, gamma=1.2):

        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in np.arange(0,256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    # ============================================
    # Adaptive Histogram Equalization
    # ============================================
    def adaptive_enhancement(self, frame):

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_enhanced = self.clahe.apply(l)

        merged = cv2.merge((l_enhanced,a,b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return enhanced

    # ============================================
    # Temporal Smoothing
    # ============================================
    def temporal_smoothing(self, frame):

        self.history_buffer.append(frame)

        avg_frame = np.mean(self.history_buffer, axis=0).astype(np.uint8)

        return avg_frame

    # ============================================
    # Frame Enhancement
    # ============================================
    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints
        """

        if enhancement_type == 'adaptive':
            enhanced = self.adaptive_enhancement(frame)

        elif enhancement_type == 'contrast':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced_gray = self.contrast_stretch(gray)
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        elif enhancement_type == 'gamma':
            enhanced = self.gamma_correction(frame)

        else:
            enhanced = frame

        # Temporal consistency
        enhanced = self.temporal_smoothing(enhanced)

        return enhanced


# ============================================
# MAIN PROGRAM
# ============================================

def main():

    print("=== REAL-TIME VIDEO ENHANCEMENT ===")
    print("Tekan tombol berikut:")
    print("1 = Adaptive Enhancement")
    print("2 = Contrast Stretching")
    print("3 = Gamma Correction")
    print("Q = Keluar")

    enhancer = RealTimeEnhancement(target_fps=30)

    # Gunakan webcam
    cap = cv2.VideoCapture(0)

    enhancement_mode = 'adaptive'

    while True:

        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Resize agar lebih cepat
        frame = cv2.resize(frame, (640,480))

        # Enhance frame
        enhanced_frame = enhancer.enhance_frame(frame, enhancement_mode)

        # Tampilkan
        combined = np.hstack((frame, enhanced_frame))

        cv2.putText(combined,
                    f"Mode: {enhancement_mode}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Real-Time Video Enhancement (Original | Enhanced)", combined)

        # Kontrol FPS
        elapsed = time.time() - start_time
        if elapsed < enhancer.frame_time:
            time.sleep(enhancer.frame_time - elapsed)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            enhancement_mode = 'adaptive'

        elif key == ord('2'):
            enhancement_mode = 'contrast'

        elif key == ord('3'):
            enhancement_mode = 'gamma'

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================
# RUN PROGRAM
# ============================================

if __name__ == "__main__":
    main()
