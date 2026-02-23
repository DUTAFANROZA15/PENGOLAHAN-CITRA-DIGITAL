# ============================================
# PRAKTIKUM 2 - LATIHAN 2: SIMULASI DIGITALISASI
# ============================================

import numpy as np
import matplotlib.pyplot as plt

def simulate_digitization(analog_signal, sampling_rate, quantization_levels):
    """
    analog_signal: fungsi kontinu f(x)
    sampling_rate: jumlah sampel per interval
    quantization_levels: jumlah level kuantisasi
    """

    # ===============================
    # 1. BUAT SINYAL ANALOG KONTINU
    # ===============================
    t_cont = np.linspace(0, 1, 1000)  # waktu kontinu
    signal_cont = analog_signal(t_cont)

    # ===============================
    # 2. SAMPLING
    # ===============================
    t_sample = np.linspace(0, 1, sampling_rate)
    signal_sample = analog_signal(t_sample)

    # ===============================
    # 3. QUANTIZATION
    # ===============================
    min_val = np.min(signal_sample)
    max_val = np.max(signal_sample)

    # Level kuantisasi
    q_levels = np.linspace(min_val, max_val, quantization_levels)

    # Kuantisasi sinyal
    signal_quantized = np.digitize(signal_sample, q_levels)
    signal_quantized = q_levels[signal_quantized - 1]

    # ===============================
    # 4. TAMPILKAN HASIL
    # ===============================
    plt.figure(figsize=(12,6))

    # Sinyal analog asli
    plt.plot(t_cont, signal_cont, label="Sinyal Analog Asli", color="blue")

    # Sinyal hasil sampling
    plt.stem(t_sample, signal_sample, linefmt="green", markerfmt="go", basefmt=" ")
    
    # Sinyal hasil kuantisasi
    plt.step(t_sample, signal_quantized, label="Sinyal Digital (Kuantisasi)", color="red")

    plt.title("Simulasi Digitalisasi: Sampling & Quantization")
    plt.xlabel("Waktu")
    plt.ylabel("Amplitudo")
    plt.legend()
    plt.grid(True)
    plt.show()

    return signal_sample, signal_quantized


# ===============================
# CONTOH FUNGSI SINYAL ANALOG
# ===============================
def analog_signal(x):
    return np.sin(2 * np.pi * 5 * x)   # sinyal sinus 5 Hz


# ===============================
# PEMANGGILAN FUNGSI
# ===============================
sampling_rate = 20        # jumlah sampel
quantization_levels = 8   # level kuantisasi

sampled, quantized = simulate_digitization(analog_signal, sampling_rate, quantization_levels)
