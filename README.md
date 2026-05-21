# MRLR-DenseNet: Skin Lesion Classification with Dynamic Multi‑scale Routing and Lightweight Residual Refinement

This repository implements **MRLR-DenseNet** – an improved DenseNet architecture for **dermoscopic skin lesion classification**, as described in the paper:

> **Fusing Dynamic Multi‑Scale Routing and Lightweight Residual Refinement for Skin Lesion Classification**  
> He Guangze, Li Yang, Yan Junfeng  
> *School of Information Science and Engineering, Hunan University of Chinese Medicine*

The model introduces two novel modules:
- **DMR (Dynamic Multi‑scale Routing)** – hierarchically extracts and adaptively fuses multi‑scale features using depthwise and dilated depthwise convolutions with spatial routing weights.
- **LRF (Lightweight Residual Refinement)** – refines intermediate feature maps with depthwise separable convolutions and residual connections, adding minimal parameters.

## 📋 Key Features

- Improved DenseNet backbone:  
  - Two consecutive 3×3 convolutions replace the initial 7×7 convolution.  
  - Each DenseLayer uses a 3×3 & 5×5 dual‑branch for local multi‑scale extraction.  
- DMR module at the end of the network performs spatially‑adaptive multi‑scale fusion.  
- LRF modules inserted after the first three DenseBlocks to strengthen edge/texture information.  
- Trained/evaluated on the **ISIC 2019** dataset (8 classes, 25,331 images, imbalanced).  
- Achieves **85.58% accuracy**, **85.66% precision**, **85.85% recall**, **85.67% F1‑score** (improving DenseNet121 by +3.69% accuracy and +4.11% F1).  
- Full training pipeline: automatic data splitting, mixup augmentation, learning rate warmup + cosine decay, Grad‑CAM visualization, confusion matrix, and classification report.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.5.1+ (with CUDA 11.8 recommended)
- Other dependencies: see `requirements.txt` below.


Place the ISIC 2019 dataset in the following structure:
isic2019/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
