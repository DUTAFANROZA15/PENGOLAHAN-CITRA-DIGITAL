# =========================================================
# SHAPE ANALYSIS PIPELINE UNTUK KLASIFIKASI OBJEK
# =========================================================
# DATASET:
# - baterai
# - korek_api
# - pensil
#
# VISUALISASI:
# 1. Visualisasi Contour Objek
# 2. Fourier Reconstruction 5 Descriptor
# 3. Fourier Reconstruction 10 Descriptor
# 4. Fourier Reconstruction 20 Descriptor
# 5. Visualisasi Chain Code
# =========================================================

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PATH DATASET
# =========================================================
DATASET_PATH = r"D:\KULIAH\SEMESTER 4\PENGOLAHAN CITRA DIGITAL\TUGAS\PERTEMUAN 11\TUGAS\PROGRAM\dataset"

CLASSES = [
    'baterai',
    'korek_api',
    'pensil'
]

# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None, None, None, None

    image = cv2.resize(image, (300, 300))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return image, gray, thresh, None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    return image, gray, thresh, contour


# =========================================================
# REGION PROPERTIES
# =========================================================
def extract_region_properties(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    M = cv2.moments(contour)

    if M['m00'] != 0:

        cx = int(M['m10'] / M['m00'])

        cy = int(M['m01'] / M['m00'])

    else:

        cx, cy = 0, 0

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = float(w) / h

    rect_area = w * h

    extent = float(area) / rect_area

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area != 0:

        solidity = float(area) / hull_area

    else:

        solidity = 0

    return {

        'area': area,
        'perimeter': perimeter,

        'centroid_x': cx,
        'centroid_y': cy,

        'bounding_x': x,
        'bounding_y': y,
        'bounding_w': w,
        'bounding_h': h,

        'aspect_ratio': aspect_ratio,

        'extent': extent,

        'solidity': solidity,

        'convex_hull_area': hull_area
    }


# =========================================================
# MOMENTS
# =========================================================
def extract_moments(contour):

    M = cv2.moments(contour)

    hu = cv2.HuMoments(M).flatten()

    hu_log = []

    for h in hu:

        value = -np.sign(h) * np.log10(abs(h) + 1e-10)

        hu_log.append(value)

    return {

        # Spatial Moments
        'm00': M['m00'],
        'm10': M['m10'],
        'm01': M['m01'],

        # Central Moments
        'mu20': M['mu20'],
        'mu02': M['mu02'],
        'mu11': M['mu11'],

        # Hu Moments
        'hu1': hu_log[0],
        'hu2': hu_log[1],
        'hu3': hu_log[2],
        'hu4': hu_log[3],
        'hu5': hu_log[4],
        'hu6': hu_log[5],
        'hu7': hu_log[6]
    }


# =========================================================
# CHAIN CODE 8 DIRECTION
# =========================================================
def get_chain_code_8(contour):

    points = contour.squeeze()

    directions = {

        (1, 0): 0,
        (1, -1): 1,
        (0, -1): 2,
        (-1, -1): 3,
        (-1, 0): 4,
        (-1, 1): 5,
        (0, 1): 6,
        (1, 1): 7
    }

    chain_code = []

    for i in range(len(points) - 1):

        dx = points[i + 1][0] - points[i][0]

        dy = points[i + 1][1] - points[i][1]

        dx = np.sign(dx)

        dy = np.sign(dy)

        direction = directions.get((dx, dy), 0)

        chain_code.append(direction)

    return chain_code


# =========================================================
# CHAIN CODE 4 DIRECTION
# =========================================================
def get_chain_code_4(contour):

    points = contour.squeeze()

    directions = {

        (1, 0): 0,
        (0, -1): 1,
        (-1, 0): 2,
        (0, 1): 3
    }

    chain_code = []

    for i in range(len(points) - 1):

        dx = points[i + 1][0] - points[i][0]

        dy = points[i + 1][1] - points[i][1]

        dx = np.sign(dx)

        dy = np.sign(dy)

        if abs(dx) > abs(dy):

            dy = 0

        else:

            dx = 0

        direction = directions.get((dx, dy), 0)

        chain_code.append(direction)

    return chain_code


# =========================================================
# NORMALISASI CHAIN CODE
# =========================================================
def normalize_chain_code(chain_code):

    if len(chain_code) == 0:
        return []

    normalized = []

    for i in range(len(chain_code) - 1):

        diff = (
            chain_code[i + 1] - chain_code[i]
        ) % 8

        normalized.append(diff)

    return normalized


# =========================================================
# POLYGON APPROXIMATION
# =========================================================
def polygon_approximation(contour):

    epsilon = 0.02 * cv2.arcLength(
        contour,
        True
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return approx


# =========================================================
# FOURIER DESCRIPTORS
# =========================================================
def fourier_descriptors(
    contour,
    num_descriptors=20
):

    contour_array = contour.squeeze()

    contour_complex = (
        contour_array[:, 0]
        + 1j * contour_array[:, 1]
    )

    fourier_result = fft(contour_complex)

    descriptors = np.abs(fourier_result)

    descriptors = descriptors[:num_descriptors]

    if len(descriptors) > 1 and descriptors[1] != 0:

        descriptors = descriptors / descriptors[1]

    return descriptors


# =========================================================
# FOURIER RECONSTRUCTION
# =========================================================
def reconstruct_contour(
    contour,
    num_descriptors
):

    contour_array = contour.squeeze()

    contour_complex = (
        contour_array[:, 0]
        + 1j * contour_array[:, 1]
    )

    fourier_result = fft(contour_complex)

    descriptors = np.copy(fourier_result)

    descriptors[num_descriptors:-num_descriptors] = 0

    reconstructed = ifft(descriptors)

    reconstructed_contour = np.array([

        [int(pt.real), int(pt.imag)]

        for pt in reconstructed

    ])

    return reconstructed_contour


# =========================================================
# VISUALISASI CONTOUR
# =========================================================
def visualize_class_contours(
    class_name,
    visual_data
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    fig.suptitle(
        f'VISUALISASI CONTOUR OBJEK - {class_name.upper()}',
        fontsize=20,
        fontweight='bold'
    )

    axes = axes.flatten()

    for i, data in enumerate(visual_data):

        image = data['image'].copy()

        contour = data['contour']

        approx = data['approx']

        filename = data['filename']

        cv2.drawContours(
            image,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        cv2.drawContours(
            image,
            [approx],
            -1,
            (0, 0, 255),
            2
        )

        axes[i].imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )

        axes[i].set_title(
            filename,
            fontsize=12
        )

        axes[i].axis('off')

    plt.tight_layout()

    plt.show()


# =========================================================
# VISUALISASI FOURIER
# =========================================================
def visualize_fourier_descriptor_page(
    class_name,
    fourier_data,
    descriptor_count
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    fig.suptitle(
        f'FOURIER DESCRIPTOR RECONSTRUCTION - {class_name.upper()} - {descriptor_count} DESCRIPTOR',
        fontsize=18,
        fontweight='bold'
    )

    axes = axes.flatten()

    for i, data in enumerate(fourier_data):

        contour = data['contour']

        filename = data['filename']

        reconstructed = reconstruct_contour(
            contour,
            descriptor_count
        )

        axes[i].plot(
            reconstructed[:, 0],
            reconstructed[:, 1]
        )

        axes[i].invert_yaxis()

        axes[i].set_title(
            filename,
            fontsize=11
        )

    plt.tight_layout()

    plt.show()


# =========================================================
# VISUALISASI CHAIN CODE
# =========================================================
def visualize_chain_code_page(
    class_name,
    chaincode_data
):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    fig.suptitle(
        f'CHAIN CODE VISUALIZATION - {class_name.upper()}',
        fontsize=18,
        fontweight='bold'
    )

    axes = axes.flatten()

    for i, data in enumerate(chaincode_data):

        image = data['image'].copy()

        contour = data['contour']

        filename = data['filename']

        chain4 = data['chain4']

        chain8 = data['chain8']

        normalized = data['normalized']

        cv2.drawContours(
            image,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        axes[i].imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )

        text_chain4 = ' '.join(
            map(str, chain4[:20])
        )

        text_chain8 = ' '.join(
            map(str, chain8[:20])
        )

        text_norm = ' '.join(
            map(str, normalized[:20])
        )

        axes[i].set_title(
            f'{filename}\n'
            f'4D : {text_chain4}\n'
            f'8D : {text_chain8}\n'
            f'NORM : {text_norm}',
            fontsize=8
        )

        axes[i].axis('off')

    plt.tight_layout()

    plt.show()


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_all_features(contour):

    region = extract_region_properties(contour)

    moments = extract_moments(contour)

    chain_8 = get_chain_code_8(contour)

    chain_4 = get_chain_code_4(contour)

    normalized_chain = normalize_chain_code(chain_8)

    fourier = fourier_descriptors(
        contour,
        20
    )

    feature_vector = {}

    feature_vector.update(region)

    feature_vector.update(moments)

    feature_vector['chain_length_8'] = len(chain_8)

    feature_vector['chain_length_4'] = len(chain_4)

    feature_vector['normalized_chain_mean'] = (

        np.mean(normalized_chain)

        if len(normalized_chain) > 0

        else 0
    )

    for i in range(len(fourier)):

        feature_vector[f'fd_{i}'] = fourier[i]

    return feature_vector


# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset():

    data = []

    for label in CLASSES:

        class_path = os.path.join(
            DATASET_PATH,
            label
        )

        contour_visual_data = []

        fourier_visual_data = []

        chaincode_visual_data = []

        print('\n' + '=' * 60)

        print(f'PROSES DATASET : {label.upper()}')

        print('=' * 60)

        for file in os.listdir(class_path):

            if file.endswith('.jpg') or file.endswith('.png'):

                image_path = os.path.join(
                    class_path,
                    file
                )

                image, gray, thresh, contour = preprocess_image(
                    image_path
                )

                if contour is None:

                    print(f'Contour gagal : {file}')

                    continue

                features = extract_all_features(
                    contour
                )

                features['label'] = label

                features['filename'] = file

                data.append(features)

                approx = polygon_approximation(
                    contour
                )

                contour_visual_data.append({

                    'image': image,

                    'contour': contour,

                    'approx': approx,

                    'filename': file
                })

                fourier_visual_data.append({

                    'contour': contour,

                    'filename': file
                })

                chain8 = get_chain_code_8(contour)

                chain4 = get_chain_code_4(contour)

                normalized = normalize_chain_code(chain8)

                chaincode_visual_data.append({

                    'image': image,

                    'contour': contour,

                    'filename': file,

                    'chain4': chain4,

                    'chain8': chain8,

                    'normalized': normalized
                })

                print(f'Berhasil memproses : {file}')

        # HALAMAN 1
        visualize_class_contours(
            label,
            contour_visual_data
        )

        # HALAMAN 2
        visualize_fourier_descriptor_page(
            label,
            fourier_visual_data,
            5
        )

        # HALAMAN 3
        visualize_fourier_descriptor_page(
            label,
            fourier_visual_data,
            10
        )

        # HALAMAN 4
        visualize_fourier_descriptor_page(
            label,
            fourier_visual_data,
            20
        )

        # HALAMAN 5
        visualize_chain_code_page(
            label,
            chaincode_visual_data
        )

    return pd.DataFrame(data)


# =========================================================
# EVALUASI FITUR
# =========================================================
def evaluate_feature_combination(
    df,
    feature_columns,
    title
):

    X = df[feature_columns]

    y = df['label']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    knn = KNeighborsClassifier(
        n_neighbors=1
    )

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print('\n' + '=' * 60)

    print(f'HASIL KLASIFIKASI : {title}')

    print('=' * 60)

    print(f'Akurasi : {accuracy * 100:.2f}%')

    print('\nConfusion Matrix')

    print(confusion_matrix(
        y_test,
        y_pred
    ))

    print('\nClassification Report')

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    return accuracy


# =========================================================
# CLASSIFICATION
# =========================================================
def classification_knn(df):

    region_features = [

        'area',
        'perimeter',
        'aspect_ratio',
        'extent',
        'solidity'
    ]

    moment_features = [

        'hu1',
        'hu2',
        'hu3'
    ]

    fourier_features = [

        'fd_1',
        'fd_2',
        'fd_3'
    ]

    all_features = (

        region_features

        + moment_features

        + fourier_features
    )

    acc_region = evaluate_feature_combination(
        df,
        region_features,
        'REGION FEATURES'
    )

    acc_moment = evaluate_feature_combination(
        df,
        moment_features,
        'MOMENT FEATURES'
    )

    acc_fourier = evaluate_feature_combination(
        df,
        fourier_features,
        'FOURIER FEATURES'
    )

    acc_all = evaluate_feature_combination(
        df,
        all_features,
        'ALL FEATURES'
    )

    print('\n' + '=' * 60)

    print('PERBANDINGAN AKURASI')

    print('=' * 60)

    print(f'Region Features  : {acc_region*100:.2f}%')

    print(f'Moment Features  : {acc_moment*100:.2f}%')

    print(f'Fourier Features : {acc_fourier*100:.2f}%')

    print(f'All Features     : {acc_all*100:.2f}%')


# =========================================================
# MAIN PROGRAM
# =========================================================
def main():

    print('=' * 60)

    print('SHAPE ANALYSIS PIPELINE UNTUK KLASIFIKASI OBJEK')

    print('=' * 60)

    df = load_dataset()

    print('\n===== DATA FITUR =====')

    print(df.head())

    df.to_csv(
        'hasil_fitur.csv',
        index=False
    )

    print('\nData fitur berhasil disimpan.')

    classification_knn(df)

    print('\nPROGRAM SELESAI')


# =========================================================
# RUN PROGRAM
# =========================================================
if __name__ == '__main__':
    main()