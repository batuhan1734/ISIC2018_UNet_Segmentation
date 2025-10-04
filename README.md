# 🩺 Skin Lesion Segmentation using U-Net (ISIC 2018 Dataset)

This project implements a **U-Net-based deep learning model** for **automatic segmentation of skin lesions** using the **ISIC 2018 Skin Lesion Segmentation Challenge dataset**.  
The goal is to accurately segment lesion areas from dermatoscopic images to assist in clinical diagnosis and research.

---

## 📘 Project Overview
Manual segmentation of skin lesions is time-consuming and prone to human error.  
This project demonstrates how **deep learning and convolutional neural networks (CNNs)** — particularly the **U-Net architecture** — can automate this process effectively.

**Key highlights:**
- Dataset: ISIC 2018 Skin Lesion Segmentation Challenge  
- Framework: PyTorch  
- Model: U-Net  
- Input size: 256×256  
- Evaluation metrics: Dice Coefficient, IoU, Precision, Recall, Confusion Matrix  

---

## 📂 Dataset

- **Source:** [ISIC 2018 Challenge Dataset](https://challenge.isic-archive.com/data#2018)  
- **License:** CC-BY-NC  
- **Citations Required:**
  - Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, *Scientific Data* 5, 180161 (2018). [DOI:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)  
  - Noel Codella et al., *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)*, 2018. [arXiv:1902.03368](https://arxiv.org/abs/1902.03368)

---

## 🧠 Model Architecture

The model is based on the **U-Net** architecture (Ronneberger et al., 2015), featuring:
- **Encoder:** Four downsampling layers with convolutional and batch normalization blocks.  
- **Decoder:** Four upsampling layers with skip connections to preserve spatial information.  
- **Output:** 1-channel sigmoid layer for binary segmentation masks.  

This structure allows the network to capture both **context** and **fine spatial details** crucial for medical image segmentation.

---

## ⚙️ Implementation Steps

### 1️⃣ Data Preparation
- Images and corresponding masks are resized to **256×256**.  
- Normalization is applied in the **[0,1]** range.  
- Augmentation techniques (flips, rotations) are used for better generalization.  
- Data is split into **80% training** and **20% validation** sets.  

*(See Colab notebook: “Data Preparation” section for code and visuals.)*

---

### 2️⃣ Model Definition
- U-Net implemented from scratch using PyTorch.  
- GPU acceleration via CUDA (Tesla T4).  

*(See “Define U-Net Model” section in the Colab notebook.)*

---

### 3️⃣ Training and Optimization
- **Loss Function:** Combined Binary Cross Entropy (BCE) + Dice Loss  
- **Optimizer:** Adam (learning rate = 1e-4)  
- **Epochs:** 10  
- **Batch Size:** 8  

Training achieved stable convergence with decreasing loss and increasing Dice/IoU values.  
*(See “Training Loop” and “Training History Graphs” sections.)*

---

### 4️⃣ Model Evaluation
- Dice Coefficient: **0.83**  
- IoU: **0.71**  
- Precision: **0.83**  
- Recall: **0.84**  

The model effectively learned lesion boundaries even with a limited subset (500 images).  
Confusion Matrix and precision–recall metrics confirmed a balanced performance between false positives and false negatives.  

*(See “Model Evaluation & Visualization” section in the notebook.)*

---

### 🔍 Example Results
The following visualizations (original image, ground truth mask, and model prediction) were generated in the Colab notebook during the evaluation phase.  
To view these examples, please open the Colab notebook linked below and scroll to the **Prediction Visualization** section.

| Example Image | Ground Truth | Predicted Mask |
|:--------------:|:-------------:|:---------------:|
| *(Available in Colab Notebook)* | *(Available in Colab Notebook)* | *(Available in Colab Notebook)* |

---

## 💬 Discussion & Insights
Even with limited data, the U-Net model achieved competitive results.  
Errors were mostly due to **boundary ambiguity**, **small lesions**, or **noise artifacts** such as hair or glare.  
Future improvements can include:
- Using full dataset (2594 images)
- Advanced augmentations (elastic deformation, color jitter)
- Loss balancing methods (Focal/Tversky Loss)
- Post-processing (morphological cleanup, CRF)
- Benchmarking against DeepLabV3 or nnU-Net

---

## 🔗 Resources

- **Google Colab Notebook:** [Add your Colab Link Here]  
- **Dataset:** [ISIC 2018 Challenge Dataset](https://challenge.isic-archive.com/data#2018)  
- **License:** CC-BY-NC  

---

## 🧾 Citation

If you use this repository, please cite:
@article{ronneberger2015unet,
title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
journal={MICCAI},
year={2015},
eprint={1505.04597},
archivePrefix={arXiv}
}


---

## 👨‍💻 Author
**[Batuhan Tunali]**  
MSc Data Analytics, BSBI  
📅 2025
