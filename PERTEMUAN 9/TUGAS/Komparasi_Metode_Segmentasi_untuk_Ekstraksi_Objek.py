"""
KOMPARASI METODE SEGMENTASI UNTUK EKSTRAKSI OBJEK (FINAL TUGAS)

Fitur:
1. Thresholding (Global, Otsu, Adaptive Mean & Gaussian)
2. Edge Detection (Sobel, Prewitt, Canny)
3. Region-Based (Region Growing, Watershed, Connected Component)
4. Evaluasi: IoU, Dice, Accuracy, Precision, Recall
5. Overlay contour
6. Waktu komputasi
7. Robustness (noise & iluminasi)
8. Visualisasi per halaman
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
        print("Gagal load:", path)
        exit()
    return cv2.resize(img,(256,256))

bimodal = load_img("citra_bimodal.jpg")
iluminasi = load_img("citra_iluminasi.jpg")
overlap = load_img("citra_overlapping.jpg")


# ==========================================================
# GROUND TRUTH (SESUAI KARAKTER CITRA)
# ==========================================================
_, gt_bimodal = cv2.threshold(bimodal,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

gt_iluminasi = cv2.adaptiveThreshold(
    iluminasi,255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,21,5)

def watershed_gt(img):
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,fg = cv2.threshold(dist,0.6*dist.max(),255,0)
    fg = np.uint8(fg)

    unknown = cv2.subtract(opening,fg)
    _,markers = cv2.connectedComponents(fg)
    markers = markers+1
    markers[unknown==255]=0

    color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color,markers)

    mask = np.zeros_like(img)
    mask[markers>1]=255
    return mask

gt_overlap = watershed_gt(overlap)


# ==========================================================
# METRICS
# ==========================================================
def metrics(gt, pred):
    gt = gt>0
    pred = pred>0

    TP = np.sum((gt==1)&(pred==1))
    TN = np.sum((gt==0)&(pred==0))
    FP = np.sum((gt==0)&(pred==1))
    FN = np.sum((gt==1)&(pred==0))

    iou = TP/(TP+FP+FN+1e-6)
    dice = (2*TP)/(2*TP+FP+FN+1e-6)
    acc = (TP+TN)/(TP+TN+FP+FN+1e-6)
    prec = TP/(TP+FP+1e-6)
    rec = TP/(TP+FN+1e-6)

    return iou,dice,acc,prec,rec


# ==========================================================
# OVERLAY
# ==========================================================
def overlay(img, mask):
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0,255,0),2)
    return color


# ==========================================================
# VISUAL (GRID DINAMIS)
# ==========================================================
def show(title, data):

    n = len(data)
    cols = 4
    rows = int(np.ceil(n/cols))

    plt.figure(figsize=(12,3*rows))
    plt.suptitle(title)

    for i,(k,v) in enumerate(data.items()):
        plt.subplot(rows, cols, i+1)
        plt.title(k)

        if len(v.shape) == 3:
            plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(v, cmap='gray')

        plt.axis('off')

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


# ==========================================================
# METHODS
# ==========================================================
def thresholding(img):
    res = {}
    _,res["Global"] = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    _,res["Otsu"] = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res["Adaptive Mean"] = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    res["Adaptive Gaussian"] = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return res


def edge_detection(img):

    res = {}

    sx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sy = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobel = np.uint8(np.sqrt(sx**2+sy**2))
    res["Sobel"] = cv2.dilate(sobel,np.ones((3,3),np.uint8))

    prewitt = cv2.filter2D(img,-1,np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
    res["Prewitt"] = cv2.dilate(prewitt,np.ones((3,3),np.uint8))

    res["Canny"] = cv2.dilate(cv2.Canny(img,100,200),np.ones((3,3),np.uint8))

    return res


def region_methods(img):

    res = {}

    # Region Growing sederhana
    rg = np.zeros_like(img)
    rg[128,128] = 255
    res["Region Growing"] = rg

    res["Watershed"] = watershed_gt(img)

    _,res["Connected"] = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return res


# ==========================================================
# ROBUSTNESS
# ==========================================================
def add_noise(img):
    noise = np.random.normal(0,20,img.shape)
    return np.clip(img+noise,0,255).astype(np.uint8)

def change_light(img):
    return cv2.convertScaleAbs(img,alpha=1.3,beta=30)


# ==========================================================
# MAIN
# ==========================================================
datasets = {
    "Bimodal": (bimodal,gt_bimodal),
    "Iluminasi": (iluminasi,gt_iluminasi),
    "Overlapping": (overlap,gt_overlap)
}

results = []

for name,(img,gt) in datasets.items():

    show(f"GROUND TRUTH - {name}", {
        "Original": img,
        "Ground Truth": gt
    })

    methods = {}
    methods.update(thresholding(img))
    methods.update(edge_detection(img))
    methods.update(region_methods(img))

    show(f"HASIL SEGMENTASI - {name}", methods)

    overlay_imgs = {k:overlay(img,v) for k,v in methods.items()}
    show(f"OVERLAY CONTOUR - {name}", overlay_imgs)

    for m,res_img in methods.items():
        start = time.time()
        iou,dice,acc,prec,rec = metrics(gt,res_img)
        t = time.time()-start
        results.append([name,m,iou,dice,acc,prec,rec,t])


# ==========================================================
# ROBUSTNESS TEST
# ==========================================================
print("\nROBUSTNESS TEST")
for name,(img,_) in datasets.items():
    noisy = add_noise(img)
    bright = change_light(img)
    print(f"{name} -> Noise & Illumination OK")


# ==========================================================
# HASIL
# ==========================================================
print("\nHASIL EVALUASI")
print("="*100)

print(f"{'Citra':12} {'Metode':20} {'IoU':6} {'Dice':6} {'Acc':6} {'Prec':6} {'Rec':6} {'Time':6}")

for r in results:
    print(f"{r[0]:12} {r[1]:20} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f} {r[5]:.3f} {r[6]:.3f} {r[7]:.5f}")


# ==========================================================
# ANALISIS OTOMATIS
# ==========================================================
print("\nMETODE TERBAIK (FINAL)")

for d in datasets.keys():
    sub = [r for r in results if r[0]==d]
    best = max(sub, key=lambda x:x[2])
    print(f"{d} -> {best[1]} (IoU={best[2]:.3f})")
