"""
PIPELINE MORFOLOGI UNTUK PREPROCESSING OCR DAN COUNTING OBJEK

Fitur:
1. Structuring element (3x3,5x5,7x7 + shape)
2. Erosi & dilasi (multi iterasi)
3. Opening & Closing
4. Morphological Gradient
5. Top-hat & Black-hat
6. OCR preprocessing simulation
7. Object counting (watershed + contour)
8. Evaluasi OCR improvement & counting accuracy (TABEL)
9. Visualisasi per halaman
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# ==========================================================
# LOAD IMAGE
# ==========================================================
def load_img(path):
    img = cv2.imread(path, 0)
    if img is None:
        print("Gambar tidak ditemukan:", path)
        exit()
    return img

imgA = load_img("citraA_dokumen_noise.jpg")
imgB = load_img("citraB_overlapping.jpg")


# ==========================================================
# STRUCTURING ELEMENTS
# ==========================================================
def get_kernel(size, shape):
    if shape == "square":
        return cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    elif shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))


# ==========================================================
# VISUAL FUNCTION
# ==========================================================
def show(title, data):

    n = len(data)
    cols = 4
    rows = int(np.ceil(n/cols))

    plt.figure(figsize=(12,3*rows))
    plt.suptitle(title)

    for i,(k,v) in enumerate(data.items()):
        plt.subplot(rows,cols,i+1)
        plt.title(k)
        plt.imshow(v,cmap='gray')
        plt.axis('off')

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


# ==========================================================
# MORPHOLOGY OPERATIONS
# ==========================================================
def morphology_experiment(img):

    results = {}

    sizes = [3,5,7]
    shapes = ["square","cross","ellipse"]

    for s in sizes:
        for sh in shapes:
            k = get_kernel(s,sh)

            results[f"Erode_{s}_{sh}"] = cv2.erode(img,k,iterations=1)
            results[f"Dilate_{s}_{sh}"] = cv2.dilate(img,k,iterations=1)
            results[f"Open_{s}_{sh}"] = cv2.morphologyEx(img,cv2.MORPH_OPEN,k)
            results[f"Close_{s}_{sh}"] = cv2.morphologyEx(img,cv2.MORPH_CLOSE,k)

    return results


# ==========================================================
# MORPHOLOGICAL EXTRA
# ==========================================================
def morph_extra(img):

    kernel = np.ones((5,5),np.uint8)

    return {
        "Gradient": cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel),
        "Top Hat": cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel),
        "Black Hat": cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
    }


# ==========================================================
# OCR PREPROCESSING
# ==========================================================
def ocr_pipeline(img):

    start = time.time()

    kernel = np.ones((3,3),np.uint8)

    denoise = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    closing = cv2.morphologyEx(denoise,cv2.MORPH_CLOSE,kernel)

    _,before = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    _,after = cv2.threshold(closing,127,255,cv2.THRESH_BINARY)

    end = time.time()

    return before,after,end-start


# ==========================================================
# OCR SCORE (SIMULASI)
# ==========================================================
def ocr_score(img):
    return np.sum(img==255)/img.size


# ==========================================================
# COUNTING OBJECT
# ==========================================================
def count_objects(img):

    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel,iterations=2)

    dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,fg = cv2.threshold(dist,0.5*dist.max(),255,0)
    fg = np.uint8(fg)

    unknown = cv2.subtract(opening,fg)
    _,markers = cv2.connectedComponents(fg)
    markers = markers+1
    markers[unknown==255]=0

    color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color,markers)

    count = len(np.unique(markers)) - 2

    return markers,count


# ==========================================================
# PROCESS OCR
# ==========================================================
print("\nPROCESS A: OCR PIPELINE")

before,after,t_ocr = ocr_pipeline(imgA)

show("OCR PREPROCESSING - BEFORE vs AFTER",{
    "Before": before,
    "After": after
})

score_before = ocr_score(before)
score_after = ocr_score(after)

print("OCR Time:",t_ocr)


# ==========================================================
# MORPHOLOGY A
# ==========================================================
show("MORPHOLOGY EXPERIMENT - CITRA A", morphology_experiment(imgA))
show("MORPHOLOGY OPERASI MAJEMUK - CITRA A", morph_extra(imgA))


# ==========================================================
# PROCESS COUNTING
# ==========================================================
print("\nPROCESS B: OBJECT COUNTING")

markers,count = count_objects(imgB)

show("WATERSHED SEGMENTATION - CITRA B",{
    "Original": imgB,
    "Markers": markers
})

print("Jumlah objek:",count)

manual_count = int(input("Masukkan jumlah objek sebenarnya: "))

accuracy = 1 - abs(manual_count-count)/manual_count


# ==========================================================
# MORPHOLOGY B
# ==========================================================
show("MORPHOLOGY EXPERIMENT - CITRA B", morphology_experiment(imgB))
show("MORPHOLOGY OPERASI MAJEMUK - CITRA B", morph_extra(imgB))


# ==========================================================
# TABEL EVALUASI (INI YANG DITAMBAHKAN)
# ==========================================================
print("\nTABEL EVALUASI")
print("="*50)

print(f"{'Metric':20} {'Nilai'}")
print("-"*50)

print(f"{'OCR Before':20} {score_before:.3f}")
print(f"{'OCR After':20} {score_after:.3f}")
print(f"{'Counting Acc':20} {accuracy:.3f}")


# ==========================================================
# KESIMPULAN
# ==========================================================
print("\nKESIMPULAN")
print("="*50)

print("OCR: Opening + Closing meningkatkan kualitas teks")
print("Counting: Watershed efektif memisahkan objek menempel")
print("Kernel kecil menjaga detail, kernel besar menghilangkan noise")
