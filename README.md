# LungDxFormer: A Transformer-CNN Hybrid Framework with Dynamic Spatial Attention for Lung Cancer Classification

## 📌 Overview
**LungDxFormer** is a deep learning framework designed for accurate and interpretable lung cancer classification from CT scan images.

The model integrates:
- Convolutional Neural Networks (CNNs) for local feature extraction
- Transformer encoders for global context modeling
- Dynamic Spatial Attention for enhanced region-focused learning

The system classifies lung nodules into:
- Benign
- Indeterminate
- Malignant

---

## 🚀 Key Features
- Hybrid CNN + Transformer architecture
- Dynamic Spatial Attention
- Multi-class classification
- End-to-end pipeline

### Explainability
- Grad-CAM
- Attention heatmaps

### Built-in
- Accuracy, Precision, Recall, F1, AUC
- Confusion Matrix & ROC

---

## 📂 Project Structure
```
LungDxFormer/
├── data/
├── configs/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   ├── utils/
│   ├── train.py
│   ├── test.py
│   └── inference.py
├── notebooks/
├── outputs/
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture
- CNN Encoder
- Positional Encoding
- Transformer Encoder
- Dynamic Spatial Attention
- Feature Fusion
- Classification Head

---

## 📊 Dataset
- LIDC-IDRI

Labels:
- 0 → Benign
- 1 → Indeterminate
- 2 → Malignant

---

## ⚙️ Installation
```bash
git clone https://github.com/your-username/LungDxFormer.git
cd LungDxFormer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Usage
```bash
python src/train.py
python src/test.py
python src/inference.py
```

---

## 📈 Metrics
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

---

## 📦 Outputs
```
outputs/
├── checkpoints/
├── metrics/
├── plots/
└── predictions/
```

---

## 📧 Contact
- Dr. K. Srinivasa Rao
- srinu532@gmail.com
