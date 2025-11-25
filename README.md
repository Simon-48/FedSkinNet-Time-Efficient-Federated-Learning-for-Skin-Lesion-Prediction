# SkinCancerBM: Benign vs. Malignant Image Classification Using CNNs, ML Models, and Federated Learning

This repository contains a structured set of experiments focused on binary image classification for skin cancer diagnosis (Benign vs. Malignant).  
The project investigates three major analytical directions:

1. **Conventional Convolutional Neural Network (CNN) models**  
2. **Classical machine-learning approaches using CNN-based feature extraction**  
3. **Federated Learning (FL) simulations across multiple clients**

The dataset employed for this work is the public Kaggle dataset **Skin Cancer: Malignant vs. Benign**, referred to in this project as **SkinCancerBM**.  
A secondary folder, `Dataset_Next`, is reserved for future datasets and extensions of this research.

---

## ▣ Project Directory Structure

```
Dataset_SkinCancerBM/
│
├── Pre_Processing/
│   ├── Cancer_vs_dataset_Pre_Process.ipynb
│   ├── Data_Spliting_Client_Ways_For_FL.ipynb
│
├── Base_Models/
│   ├── Cancer_vs_EfficientNetB0_FL_Project_2.ipynb
│   ├── Cancer_vs_InceptionResNetV2_FL_Project_2.ipynb
│   ├── Cancer_vs_InceptionV3_FL_Project_2.ipynb
│   ├── Cancer_vs_MobileNetV2_FL_Project_2.ipynb
│   ├── Cancer_vs_NASNetLarge_FL_Project_2.ipynb
│   ├── Cancer_vs_ResNet50_FL_Project_2.ipynb
│   ├── Cancer_vs_VGG19_FL_Project_2.ipynb
│   ├── Cancer_vs_XCEPTION_FL_Project_2.ipynb
│
├── ML_Based_Models/
│   ├── Cancer_vs_ML_EfficientNetB0_FL_Project_2.ipynb
│   ├── Cancer_vs_ML_ResNet50_FL_Project_2.ipynb
│   ├── Cancer_vs_ML_VGG19_FL_Project_2.ipynb
│
└── Federated_Learning/
    ├── FL_DenseNet201_FLWR.ipynb
    ├── FL_EfficientNetB0_FLWR.ipynb
    ├── FL_Inception_V3_FLWR.ipynb
    ├── FL_MobileNet_V2_FLWR.ipynb
    ├── FL_ResNet50_FLWR.ipynb
    ├── FL_VGG19_FLWR.ipynb

Dataset_Next/
```




---

## ▣ 1. Pre-Processing

Located in: `Dataset_SkinCancerBM/Pre_Processing/`

### ● Cancer_vs_dataset_Pre_Process.ipynb  
Performs essential preprocessing including:
- image resizing  
- normalization  
- metadata organization  
- dataset consistency verification  

No augmentation procedures are applied in this phase.

### ● Data_Spliting_Client_Ways_For_FL.ipynb  
Implements **stratified client-wise splitting** for federated learning simulations.  
The data are distributed across **10 clients**, ensuring balanced class proportions for Benign and Malignant cases.

---

## ▣ 2. Baseline CNN Models

Located in: `Dataset_SkinCancerBM/Base_Models/`

This section explores several well-known CNN architectures as baseline supervised classifiers.  
All models are used **without transfer learning, fine-tuning, or augmentation**.  
Each notebook contains:
- dataset loading  
- simple preprocessing  
- model training from scratch  
- evaluation metrics

Models examined:
- EfficientNetB0  
- InceptionResNetV2  
- InceptionV3  
- MobileNetV2  
- NASNetLarge  
- ResNet50  
- VGG19  
- Xception  

These results serve as reference points for comparison with ML-based and federated approaches.

---

## ▣ 3. Machine-Learning Models with CNN Feature Extraction

Located in: `Dataset_SkinCancerBM/ML_Based_Models/`

Here, deep CNNs are employed solely as **feature extractors**.  
Extracted embeddings (via pooling operations) are then used to train ML classifiers.  
XGBoost is used as the primary classical machine learning model.

Implemented pipelines:
- EfficientNetB0 → XGBoost  
- ResNet50 → XGBoost  
- VGG19 → XGBoost  

This approach allows a contrast between end-to-end CNN training and hybrid deep-feature + ML classification strategies.

---

## ▣ 4. Federated Learning Experiments (FLWR + PyTorch)

Located in: `Dataset_SkinCancerBM/Federated_Learning/`

Federated Learning experiments are conducted using the **Flower (FLWR)** framework.  
The dataset is partitioned into **10 stratified clients**, and each client performs local training.  
The global model is updated via FedAvg aggregation.

Models studied:
- DenseNet201  
- EfficientNetB0  
- InceptionV3  
- MobileNetV2  
- ResNet50  
- VGG19  

These notebooks explore the behaviour of CNN training in decentralized settings, providing insights into performance differences relative to centralized training.

---

## ▣ Dataset Description

**Dataset:** Skin Cancer — Malignant vs. Benign  
**Short identifier:** *SkinCancerBM*  
Includes two image classes:
- Benign  
- Malignant  

Images vary in resolution and acquisition conditions, requiring normalization prior to analysis.

---

## ▣ Future Work

The `Dataset_Next/` directory is reserved for forthcoming research, including:
- evaluation on additional dermatology datasets  
- experiments with Vision Transformers (ViT, Swin, ConvNeXt)  
- advanced federated learning strategies (e.g., FedProx, FedNova)  
- privacy-preserving FL (secure aggregation, differential privacy)  
- improved ML models with hyperparameter optimization  

---

## ▣ Acknowledgments  
This research utilizes the public dataset available on Kaggle and builds upon open-source deep learning and federated learning frameworks.