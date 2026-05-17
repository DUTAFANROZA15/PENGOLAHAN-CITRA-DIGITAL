import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import warnings
warnings.filterwarnings("ignore")

import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATH DATASET
# =========================================================

DATASET_PATH = r"D:\KULIAH\SEMESTER 4\PENGOLAHAN CITRA DIGITAL\TUGAS\PERTEMUAN 12\TUGAS\PROGRAM\dataset"

# =========================================================
# MEMBACA DATASET
# =========================================================


def load_dataset(dataset_path):
    images = []
    labels = []
    paths = []

    classes = os.listdir(dataset_path)

    for class_name in classes:

        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):

            if file.endswith((".jpg", ".png", ".jpeg")):

                image_path = os.path.join(class_path, file)

                image = cv2.imread(image_path)

                if image is not None:
                    images.append(image)
                    labels.append(class_name)
                    paths.append(image_path)

    return images, labels, paths

# =========================================================
# FEATURE DETECTOR
# =========================================================


def get_sift():
    return cv2.SIFT_create()



def get_orb():
    return cv2.ORB_create(nfeatures=1000)



def get_surf():

    try:
        return cv2.xfeatures2d.SURF_create()

    except:
        return None

# =========================================================
# EKSTRAKSI FITUR
# =========================================================


def extract_features(detector, image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start = time.time()

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    end = time.time()

    extraction_time = end - start

    return keypoints, descriptors, extraction_time

# =========================================================
# VISUALISASI KEYPOINTS DALAM SATU HALAMAN
# =========================================================


def visualize_keypoints_grid(images, labels, detector, method_name):

    unique_labels = []
    selected_images = []

    for image, label in zip(images, labels):

        if label not in unique_labels:
            unique_labels.append(label)
            selected_images.append((image, label))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, (image, label) in enumerate(selected_images):

        kp, desc, _ = extract_features(detector, image)

        result = cv2.drawKeypoints(
            image,
            kp,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'{method_name} - {label}')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Visualisasi Keypoints {method_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

# =========================================================
# BRUTE FORCE MATCHING
# =========================================================


def bf_matching(desc1, desc2, method='SIFT'):

    if desc1 is None or desc2 is None:
        return []

    if method == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

# =========================================================
# FLANN MATCHING
# =========================================================


def flann_matching(desc1, desc2, method='SIFT'):

    if desc1 is None or desc2 is None:
        return []

    if method == 'ORB':

        index_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )

        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        desc1 = np.uint8(desc1)
        desc2 = np.uint8(desc2)

    else:

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

# =========================================================
# RANSAC HOMOGRAPHY
# =========================================================


def ransac_homography(kp1, kp2, matches):

    if len(matches) < 4:
        return None, None

    src_pts = np.float32([
        kp1[m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    return H, mask

# =========================================================
# VISUALISASI FEATURE MATCHING DALAM SATU HALAMAN
# =========================================================


def draw_matching_grid(images, labels, detector, method='SIFT', matching_type='BF'):

    unique_labels = []
    selected_pairs = []

    for i in range(len(images) - 1):

        if labels[i] not in unique_labels:

            unique_labels.append(labels[i])

            selected_pairs.append((
                images[i],
                images[i + 1],
                labels[i]
            ))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, (img1, img2, label) in enumerate(selected_pairs):

        kp1, desc1, _ = extract_features(detector, img1)
        kp2, desc2, _ = extract_features(detector, img2)

        if matching_type == 'BF':
            matches = bf_matching(desc1, desc2, method)
        else:
            matches = flann_matching(desc1, desc2, method)

        result = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches[:30],
            None,
            flags=2
        )

        axes[idx].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f'{matching_type} - {label}')
        axes[idx].axis('off')

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'{matching_type} Matching SIFT', fontsize=16)
    plt.tight_layout()
    plt.show()

# =========================================================
# BAG OF VISUAL WORDS
# =========================================================


def build_vocabulary(descriptor_list, k=50):

    descriptors = np.vstack(descriptor_list)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(descriptors)

    return kmeans

# =========================================================
# HISTOGRAM VISUAL WORDS
# =========================================================


def build_histogram(descriptors, kmeans):

    histogram = np.zeros(len(kmeans.cluster_centers_))

    if descriptors is None:
        return histogram

    predictions = kmeans.predict(descriptors)

    for p in predictions:
        histogram[p] += 1

    return histogram

# =========================================================
# EKSTRAKSI BOVW FEATURES
# =========================================================


def extract_bovw_features(images, detector, kmeans):

    features = []

    for image in images:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = detector.detectAndCompute(gray, None)

        hist = build_histogram(desc, kmeans)

        features.append(hist)

    return np.array(features)

# =========================================================
# PCA REDUCTION
# =========================================================


def apply_pca(descriptors, n_components):

    pca = PCA(n_components=n_components)

    reduced = pca.fit_transform(descriptors)

    return reduced, pca

# =========================================================
# EVALUASI PCA + GRAFIK
# =========================================================


def evaluate_pca(descriptor_matrix, method_name):

    components = [16, 32, 64, 128]

    compression_values = []
    valid_components = []

    print("\nEVALUASI PCA")
    print("=" * 50)

    for comp in components:

        if descriptor_matrix.shape[1] >= comp:

            reduced, _ = apply_pca(descriptor_matrix, comp)

            compression = (
                comp / descriptor_matrix.shape[1]
            ) * 100

            compression_values.append(compression)
            valid_components.append(comp)

            print(f"Komponen PCA : {comp}")
            print(f"Ukuran Asli  : {descriptor_matrix.shape}")
            print(f"Ukuran Baru  : {reduced.shape}")
            print(f"Kompresi     : {compression:.2f}%")
            print("-" * 40)

    plt.figure(figsize=(8, 5))

    plt.plot(
        valid_components,
        compression_values,
        marker='o'
    )

    plt.title(
        f'Pengaruh PCA Components terhadap Kompresi ({method_name})'
    )

    plt.xlabel('Jumlah PCA Components')
    plt.ylabel('Kompresi (%)')
    plt.grid(True)
    plt.show()

# =========================================================
# TABEL RINGKASAN
# =========================================================


def create_summary_table(df):

    summary = df.groupby('Method').agg({
        'Keypoints': 'mean',
        'Extraction Time': 'mean',
        'Descriptor Dimension': 'mean'
    }).reset_index()

    print("\nTABEL RINGKASAN METODE")
    print("=" * 60)
    print(summary)

# =========================================================
# MAIN PROGRAM
# =========================================================


def main():

    print("=" * 60)
    print("SISTEM PENCOCOKAN OBJEK BERBASIS FITUR LOKAL")
    print("=" * 60)

    # =====================================================
    # LOAD DATASET
    # =====================================================

    images, labels, paths = load_dataset(DATASET_PATH)

    print(f"Jumlah citra : {len(images)}")

    # =====================================================
    # DETECTOR
    # =====================================================

    detectors = {
        'SIFT': get_sift(),
        'ORB': get_orb()
    }

    surf = get_surf()

    if surf is not None:
        detectors['SURF'] = surf

    # =====================================================
    # EKSTRAKSI FITUR
    # =====================================================

    results = []

    for method_name, detector in detectors.items():

        print(f"\nMETODE : {method_name}")
        print("-" * 50)

        all_descriptors = []

        for image, label, path in zip(images, labels, paths):

            kp, desc, ext_time = extract_features(
                detector,
                image
            )

            if desc is not None:
                all_descriptors.append(desc)

            desc_dim = 0

            if desc is not None:
                desc_dim = desc.shape[1]

            print(f"Objek              : {label}")
            print(f"File               : {os.path.basename(path)}")
            print(f"Jumlah Keypoints   : {len(kp)}")
            print(f"Dimensi Descriptor : {desc_dim}")
            print(f"Waktu Ekstraksi    : {ext_time:.4f} detik")
            print("-" * 30)

            results.append([
                method_name,
                label,
                os.path.basename(path),
                len(kp),
                desc_dim,
                ext_time
            ])

        # =================================================
        # VISUALISASI KEYPOINTS
        # =================================================

        visualize_keypoints_grid(
            images,
            labels,
            detector,
            method_name
        )

        # =================================================
        # PCA
        # =================================================

        try:

            descriptor_matrix = np.vstack(all_descriptors)

            evaluate_pca(
                descriptor_matrix,
                method_name
            )

        except:
            print("PCA gagal diproses")

    # =====================================================
    # DATAFRAME HASIL
    # =====================================================

    df = pd.DataFrame(results, columns=[
        'Method',
        'Object',
        'File',
        'Keypoints',
        'Descriptor Dimension',
        'Extraction Time'
    ])

    print("\nTABEL HASIL EKSTRAKSI")
    print(df)

    create_summary_table(df)

    # =====================================================
    # FEATURE MATCHING
    # =====================================================

    print("\nFEATURE MATCHING")
    print("=" * 50)

    sift = get_sift()

    reference_image = images[0]
    test_image = images[1]

    kp1, desc1, _ = extract_features(
        sift,
        reference_image
    )

    kp2, desc2, _ = extract_features(
        sift,
        test_image
    )

    # =====================================================
    # BRUTE FORCE
    # =====================================================

    bf_matches = bf_matching(desc1, desc2, 'SIFT')

    print(f"Jumlah BF Matches : {len(bf_matches)}")

    draw_matching_grid(
        images,
        labels,
        sift,
        method='SIFT',
        matching_type='BF'
    )

    # =====================================================
    # FLANN
    # =====================================================

    flann_matches = flann_matching(desc1, desc2, 'SIFT')

    print(f"Jumlah FLANN Matches : {len(flann_matches)}")

    draw_matching_grid(
        images,
        labels,
        sift,
        method='SIFT',
        matching_type='FLANN'
    )

    # =====================================================
    # RANSAC
    # =====================================================

    H, mask = ransac_homography(
        kp1,
        kp2,
        flann_matches
    )

    if mask is not None:

        inliers = np.sum(mask)
        outliers = len(mask) - inliers

        print(f"Jumlah Inliers  : {inliers}")
        print(f"Jumlah Outliers : {outliers}")

    # =====================================================
    # PRECISION RECALL CURVE
    # =====================================================

    distances = [m.distance for m in flann_matches]

    if len(distances) > 0:

        y_true = np.ones(len(distances))
        y_scores = np.max(distances) - np.array(distances)

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_scores
        )

        ap = average_precision_score(y_true, y_scores)

        plt.figure(figsize=(8, 5))
        plt.plot(recall, precision)
        plt.title(f'Precision Recall Curve (AP={ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.show()

    # =====================================================
    # BAG OF VISUAL WORDS
    # =====================================================

    print("\nBAG OF VISUAL WORDS")
    print("=" * 50)

    sift = get_sift()

    descriptor_list = []

    for image in images:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = sift.detectAndCompute(gray, None)

        if desc is not None:
            descriptor_list.append(desc)

    vocabulary_sizes = [10, 20, 50, 100]

    accuracies = []
    confusion_results = []

    for vocab_size in vocabulary_sizes:

        print(f"\nVocabulary Size : {vocab_size}")

        kmeans = build_vocabulary(
            descriptor_list,
            vocab_size
        )

        features = extract_bovw_features(
            images,
            sift,
            kmeans
        )

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.3,
            random_state=42
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm = SVC(kernel='linear')

        svm.fit(X_train, y_train)

        predictions = svm.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        accuracies.append(accuracy * 100)

        print(f"Akurasi SVM : {accuracy * 100:.2f}%")

        cm = confusion_matrix(y_test, predictions)

        confusion_results.append((cm, vocab_size))

        print(classification_report(y_test, predictions))

    # =====================================================
    # CONFUSION MATRIX DALAM SATU HALAMAN
    # =====================================================

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (cm, vocab_size) in enumerate(confusion_results):

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=np.unique(labels)
        )

        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f'Vocabulary Size = {vocab_size}')

    plt.suptitle('Confusion Matrix BoVW + SVM', fontsize=16)
    plt.tight_layout()
    plt.show()

    # =====================================================
    # GRAFIK VOCABULARY SIZE
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        vocabulary_sizes,
        accuracies,
        marker='o'
    )

    plt.title('Pengaruh Vocabulary Size terhadap Akurasi')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Akurasi (%)')
    plt.grid(True)
    plt.show()

    # =====================================================
    # ANALISIS HASIL
    # =====================================================

    print("\nANALISIS HASIL")
    print("=" * 50)

    print("1. SIFT memiliki akurasi matching paling stabil")
    print("2. ORB memiliki proses paling cepat")
    print("3. FLANN lebih efisien dibanding brute-force")
    print("4. RANSAC mampu mengurangi outlier matching")
    print("5. Vocabulary size besar meningkatkan akurasi")
    print("6. PCA mampu mengurangi dimensi descriptor")
    print("7. SIFT cocok untuk rotasi dan skala")
    print("8. ORB cocok untuk aplikasi real-time")
    print("9. SIFT + FLANN + RANSAC cocok untuk object recognition")
    print("10. ORB cocok untuk mobile vision")
    print("11. BoVW + SVM cocok untuk klasifikasi multi-objek")
    print("12. PCA cocok untuk kompresi descriptor")

    print("\nPROGRAM SELESAI")

# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == '__main__':
    main()