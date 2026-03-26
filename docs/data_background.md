# Dataset Background: PhysioNet/CinC Challenge 2017

## Overview

This project uses the dataset from the **PhysioNet Computing in Cardiology Challenge 2017**, which focuses on automatic classification of short single-lead ECG recordings. The challenge goal is to distinguish between four rhythm classes to support clinical AF detection from patient-initiated, brief ECG recordings captured by wearable devices.

- **Source**: [PhysioNet Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/)
- **Device**: AliveCor single-lead ECG recorder
- **Sampling Rate**: 300 Hz
- **Recording Length**: 9–61 seconds (mean ≈ 32.5 seconds)
- **Signal Processing**: Band-pass filtered by the AliveCor device

---

## Classification Task

Each ECG recording is assigned one of four labels:

| Label | Class | Description |
|-------|-------|-------------|
| `N` | Normal | Normal sinus rhythm |
| `A` | AF | Atrial fibrillation |
| `O` | Other | Other rhythm (not normal, not AF) |
| `~` | Noisy | Recording too noisy to classify |

---

## Dataset Split

| Split | Records |
|-------|---------|
| Training | 8,528 |
| Validation (subset of training) | 300 |
| Test (hidden) | 3,658 |

---

## Class Distribution (Training Set)

| Class | Count | Mean Duration (s) |
|-------|-------|-------------------|
| Normal | 5,154 | 31.9 |
| AF | 771 | 31.6 |
| Other | 2,557 | 34.1 |
| Noisy | 46 | 27.1 |
| **Total** | **8,528** | |

> **Note**: The dataset is heavily imbalanced. AF accounts for only ~9% of training samples, and Noisy for less than 1%. This must be addressed during model training (e.g., class weighting, oversampling, or augmentation).

---

## Data Format

- **Signal file**: `.mat` (MATLAB V4, WFDB-compliant) — contains the raw ECG signal as a 1D array
- **Header file**: `.hea` — contains metadata (sampling rate, signal length, etc.)
- **Label file**: `REFERENCE.csv` — maps each recording filename to its class label

---

## Evaluation Metric

Performance is measured by the **mean F1-score** across all four classes:

$$F_1 = \frac{F_{1,\text{Normal}} + F_{1,\text{AF}} + F_{1,\text{Other}} + F_{1,\text{Noisy}}}{4}$$

This metric penalizes poor performance on any single class, including the minority classes (AF and Noisy).

---

## Clinical Motivation

Atrial fibrillation is the most common sustained cardiac arrhythmia, affecting millions worldwide and significantly increasing the risk of stroke. Early and accurate detection from short, single-lead recordings enables timely clinical intervention, making this classification task clinically impactful.
