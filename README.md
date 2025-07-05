# MULTI-MODEL-HYBRID-ML-DL-FRAMEWORK-FOR-EARLY-DETECTION-OF-SKIN-CANCER

[![Python version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## 🧠 Overview

This project presents a **hybrid ML–DL framework** for early detection of skin cancer. It combines pretrained CNN feature extractors (ResNet-50, EfficientNet-B0, InceptionV3) with classical ML classifiers (SVM, KNN, Random Forest, XGBoost) to detect 7 types of skin lesions from dermoscopic images using the **HAM10000** dataset.

> ✅ Achieved **95.92% accuracy** with EfficientNet-B0 + XGBoost on test data (state-of-the-art performance in this pipeline).

---

## 🚀 Key Features

- ✅ **Hybrid ML-DL Architecture**
- 🧠 **CNNs**: ResNet-50, InceptionV3, EfficientNet-B0
- 🧪 **Classical ML Classifiers**: SVM, KNN, Random Forest, XGBoost
- 📊 **Metrics**: Accuracy, Recall, Precision, F1-Score
- 📁 **Dataset**: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- 💻 **Kaggle Notebook Compatible** (GPU supported)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/MULTI-MODEL-HYBRID-ML-DL-FRAMEWORK-FOR-EARLY-DETECTION-OF-SKIN-CANCER.git
cd MULTI-MODEL-HYBRID-ML-DL-FRAMEWORK-FOR-EARLY-DETECTION-OF-SKIN-CANCER
pip install -r requirements.txt
