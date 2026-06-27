# MRLR-DenseNet: Fusing Dynamic Multi‑Scale Routing and Lightweight Residual Refinement for Skin Lesion Classification

This repository provides the official PyTorch implementation of **MRLR-DenseNet**, an improved DenseNet architecture for dermoscopic skin lesion classification, as proposed in the paper:

> **Fusing Dynamic Multi‑Scale Routing and Lightweight Residual Refinement for Skin Lesion Classification**  
> He Guangze, Li Yang, Yan Junfeng  
> *School of Information Science and Engineering, Hunan University of Chinese Medicine*  
> *AI TCM Lab Hunan, Changsha, China*

The model introduces three key improvements over standard DenseNet:
1. **Improved DenseLayer** – Replaces the single 3×3 convolution with a dual‑branch (3×3 and 5×5) structure for enhanced local multi‑scale feature extraction.
2. **Lightweight Residual Refinement (LRF)** – Inserted after the first three DenseBlocks, these modules refine intermediate features with depthwise separable convolutions and a residual connection, preserving edge/texture details with minimal parameters.
3. **Dynamic Multi‑scale Routing (DMR)** – Placed at the end of the backbone, this module generates hierarchical multi‑scale responses (using 5×5, 7×7 dilated, and 9×9 dilated depthwise convolutions) and fuses them via spatially adaptive routing weights.

The model is evaluated on the **ISIC 2019** dataset (8 classes, 25,331 images, imbalanced) and achieves **85.58% accuracy**, **85.66% precision**, **85.85% recall**, and **85.67% F1‑score**, outperforming DenseNet121 by +3.69% accuracy and +4.11% F1‑score.


##  Model Architecture Overview

- **Backbone**: DenseNet121
- **Initial Layer**: Two consecutive 3×3 convolutions (instead of the original 7×7 conv)
- **DenseBlock**: Each DenseLayer uses a 3×3 + 5×5 dual‑branch (outputs added element‑wise)
- **LRF Modules**: Inserted after the first three DenseBlocks (before each transition layer)
- **DMR Module**: Applied after the final DenseBlock and batch normalization – performs hierarchical multi‑scale routing and spatial recalibration
- **Classifier**: Global average pooling + fully connected layer

For detailed architecture diagrams, please refer to the original paper.

##  Dataset Information

- **Dataset**: ISIC 2019 Skin Lesion Classification Challenge
- **Source**: [ISIC 2019 on Kaggle](https://www.kaggle.com/datasets/interestingstats/isic2019-dataset/data)
- **Total Images**: 25,331
- **Classes**: 8 – Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Actinic Keratosis (AK), Benign Keratosis (BKL), Dermatofibroma (DF), Vascular Lesion (VASC), Squamous Cell Carcinoma (SCC)
- **Class Distribution**: Imbalanced (NV: 12,875; SCC: 327)
- **Split**: 8:2 stratified random split (20,179 train / 5,152 test)
  
**Please download the dataset from the provided source link, as large files cannot be uploaded to this repository.**


##  Installation

git clone https://github.com/shuiling123/MRLR_Densenet.git
cd MRLR_Densenet

pip install -r requirements.txt

##  Training
python train.py


##  Data Preparation

isic2019/
├── train/
│   ├── MEL/
│   ├── NV/
│   ├── BCC/
│   ├── AK/
│   ├── BKL/
│   ├── DF/
│   ├── VASC/
│   └── SCC/
└── test/               
    └── (same class subfolders)



##  Citation

@article{codella2019skin,
  title={Skin lesion analysis toward melanoma detection 2019: A challenge hosted by the international skin imaging collaboration (ISIC)},
  author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and Celebi, M Emre and Dusza, Stephen and Gutman, David and Helba, Brian and Kalloo, Aadi and Liopyris, Konstantinos and Marchetti, Michael and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}


##  License

Code: This project is licensed under the MIT License.

Dataset: The ISIC 2019 dataset is publicly available for non-commercial research use under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/).
