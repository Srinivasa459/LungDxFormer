LungDxFormer: A Transformer-CNN Hybrid Framework with Dynamic Spatial Attention for Lung Cancer Classification
📌 Overview
LungDxFormer is a deep learning framework designed for accurate and interpretable lung cancer classification from CT scan images. The model integrates convolutional neural networks (CNNs) for local feature extraction with Transformer encoders for global context modeling, enhanced by a dynamic spatial attention mechanism.
The system classifies lung nodules into three clinically relevant categories:
•	Benign
•	Indeterminate
•	Malignant
This repository provides a complete, modular, and reproducible implementation of the proposed framework.
________________________________________
🚀 Key Features
•	Hybrid CNN + Transformer architecture
•	Dynamic Spatial Attention for region-focused learning
•	Multi-class classification with softmax output
•	End-to-end pipeline: preprocessing → training → inference
•	Explainability using:
o	Grad-CAM
o	Attention heatmaps
•	Built-in:
o	Evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
o	Confusion matrix & ROC curves
o	Ablation study support
•	Modular and scalable design for research use
________________________________________
📂 Project Structure
LungDxFormer/
│
├── data/                  # Raw and processed datasets
├── configs/               # Configuration files
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model components
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Metrics and evaluation
│   ├── explainability/    # Grad-CAM and attention maps
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   ├── test.py            # Testing script
│   └── inference.py       # Inference script
│
├── notebooks/             # Jupyter notebooks
├── outputs/               # Results, checkpoints, plots
├── requirements.txt
└── README.md
________________________________________
🧠 Model Architecture
The proposed LungDxFormer consists of the following components:
1.	CNN Encoder
o	Extracts local spatial features from CT ROI patches
2.	Positional Encoding
o	Adds spatial awareness to Transformer inputs
3.	Transformer Encoder
o	Captures global dependencies using multi-head self-attention
4.	Dynamic Spatial Attention
o	Focuses on diagnostically relevant regions
5.	Feature Fusion Layer
o	Combines CNN features and attention-refined Transformer outputs
6.	Classification Head
o	Outputs class probabilities via softmax
________________________________________
📊 Dataset
The framework is designed for use with:
•	LIDC-IDRI Dataset (Lung CT scans with annotations)
Expected Data Format
•	Preprocessed ROI patches (e.g., 64×64 / 96×96)
•	Labels mapped to:
o	0 → Benign
o	1 → Indeterminate
o	2 → Malignant
⚠️ Note: Dataset is not included due to licensing restrictions.
________________________________________
⚙️ Installation
Step 1: Clone Repository
git clone https://github.com/your-username/LungDxFormer.git
cd LungDxFormer
Step 2: Create Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Step 3: Install Dependencies
pip install -r requirements.txt
________________________________________
▶️ Usage
🔹 Train the Model
python src/train.py
🔹 Evaluate the Model
python src/test.py
🔹 Run Inference
python src/inference.py
________________________________________
🧪 Quick Test (Synthetic Data)
A synthetic dataset generator is included to verify the pipeline:
python src/data/dataset.py
This allows testing the full training pipeline without real medical data.
________________________________________
📈 Evaluation Metrics
The model is evaluated using:
•	Accuracy
•	Precision (per class & macro)
•	Recall (per class & macro)
•	F1-score
•	ROC-AUC (one-vs-rest)
•	Confusion Matrix
________________________________________
🔍 Explainability
The framework provides interpretability using:
•	Grad-CAM for CNN feature visualization
•	Spatial Attention Maps for Transformer attention regions
Outputs include:
•	Heatmaps overlaid on CT ROI images
•	Class prediction confidence
________________________________________
🔬 Ablation Study
The repository supports controlled experiments by enabling/disabling:
•	Transformer module
•	Spatial attention
•	Positional encoding
•	Fusion strategies
Modify config settings in:
configs/config.yaml
________________________________________
📦 Outputs
After training/testing, outputs are saved in:
outputs/
├── checkpoints/
├── metrics/
├── plots/
└── predictions/
________________________________________
🧾 Reproducibility
•	Fixed random seeds for consistency
•	Modular configuration-driven pipeline
•	Patient-level dataset splitting recommended
•	Hyperparameters defined in config files
________________________________________
⚠️ Limitations
•	LIDC preprocessing may require dataset-specific adjustments
•	Performance depends on ROI extraction quality
•	Transformer complexity may increase training time
________________________________________
🔮 Future Work
•	3D or 2.5D CT modeling
•	Multi-modal fusion (clinical + imaging data)
•	Self-supervised pretraining
•	Lightweight deployment for edge devices
________________________________________
🤝 Contributing
Contributions are welcome. Please:
1.	Fork the repository
2.	Create a new branch
3.	Submit a pull request
________________________________________
📜 License
This project is for research and academic use. Please cite appropriately if used in publications.
________________________________________
📧 Contact
For research collaboration or queries:
•	Author: [Your Name]
•	Email: [Your Email]
________________________________________
⭐ Acknowledgment
This work is inspired by recent advances in hybrid deep learning architectures combining CNNs, Transformers, and attention mechanisms for medical image analysis.

