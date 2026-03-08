# 🏥 AI Chest X-ray Classification Project: A Hybrid Deep Learning Study

This research project presents a multi-phase exploration into optimizing multi-label thoracic pathology detection using the **NIH ChestX-ray14** dataset. We transition from standard CNN benchmarks to a state-of-the-art **CNN-Transformer Hybrid** architecture.

---

## 🔬 Abstract
Medical imaging requires both local feature precision (for small nodules) and global context (for lung-wide pathologies). Our study demonstrates that fusing **DenseNet121** with **Swin Transformers** at high resolutions (**384x384**) significantly outperforms traditional methods. By implementing **Spatial Attention** and **Patient-wise splitting**, we achieve a final **Macro AUC of 0.884**.

---

## 📈 Research Benchmark: Phase-wise Evolution
We conducted four major phases of experimentation to reach the final optimized pipeline:

| Phase | Model Architecture | Image Size | Loss Function | Macro AUC | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P1** | CheXNet (DenseNet121) | 224x224 | BCE | 0.843 | 0.144 | Baseline |
| **P2** | Novel Hybrid (CNN+Swin) | 224x224 | Focal | 0.866 | 0.234 | Improved |
| **P3** | Novel Hybrid (CNN+Swin) | 384x384 | Focal | 0.869 | 0.248 | Scaled |
| **P4** | **Hybrid BCE Optimized** | **384x384** | **BCE** | **0.884** | **0.278** | **Best** |

---

## 🏗 Detailed System Architecture

### 1. Dual-Branch Feature Extraction
- **Branch A (CNN)**: Uses a DenseNet121 backbone to extract high-resolution spatial textures. It focuses on local anatomical details.
- **Branch B (Transformer)**: Uses a Swin Transformer (Tiny) to model long-range dependencies across the entire lung field.
- **Spatial Attention Layer**: A custom module that computes attention weights along the spatial dimensions, helping the CNN branch ignore non-lung regions.

### 2. Information Fusion & Classification
- **Fusion Layer**: Concatenates results from both branches into a single 1792-dimensional feature vector.
- **MLP Head**: A multi-layer perceptron (Linear -> ReLU -> Dropout) that interprets the fused features.
- **Multi-Label Output**: Final Sigmoid-activated layer predicting probabilities for 14 individual pathologies.

### 3. Training Optimization
- **CLAHE Preprocessing**: Applied to enhance local contrast in grayscale X-rays.
- **Data Integrity**: Implemented a **Patient-wise Split** (80/10/10) to ensure that images from the same patient never exist in both training and test sets, preventing artificial performance inflation (leakage).
- **Optimizer**: AdamW with Cosine Annealing learning rate schedule.

---

## 📂 Repository Key Directories

- `hybrid_384_bce_experiment/`: Final optimized experiment with robust data splitting and BCE loss.
- `all_model_weights/`: Centralized storage for all trained model checkpoints (.pth files).
- `comparisons_and_results/`: Centralized storage for all CSV benchmarking data.
- `project_documentation/`: Premium web-based project report and visualizers.
- `chexnet_baseline/`: Initial implementation using DenseNet121.
- `novel_pipeline_224/`: First iteration of the hybrid CNN-Transformer model.
- `novel_pipeline_384_highres/`: Scaling the hybrid model to 384px resolution.

---

##  Performance by Pathology (Best Model)
Selected metrics for our top-performing classes using Phase 4:

| Pathology | AUC Score | F1-Score | Specificity |
| :--- | :--- | :--- | :--- |
| **Emphysema** | 0.963 | 0.561 | 0.992 |
| **Cardiomegaly** | 0.955 | 0.343 | 0.998 |
| **Pneumothorax** | 0.939 | 0.468 | 0.992 |
| **Effusion** | 0.924 | 0.516 | 0.982 |
| **Hernia** | 0.923 | 0.000 | 1.000 |

---

## 🛠 Getting Started

### Prerequisites
```bash
pip install torch torchvision timm pandas numpy opencv-python scikit-learn pillow tqdm
```

### Running the Best Pipeline
```bash
python3 hybrid_384_bce_experiment/hybrid_bce_pipeline.py
```

### Viewing the Visual Dashboard
Open the following file in any modern web browser to view interactive charts and pipeline diagrams:
`project_documentation/index.html`

---

## � Technical Glossary
- **Swin Transformer**: Hierarchical ViT using shifted windows for efficient memory usage.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization.
- **Patient-wise Split**: Splitting strategy based on PatientID to prevent data leakage.

---
*Developed for Advanced Medical AI Research 2026. This project is optimized for Apple Silicon (MPS) and CUDA-capable environments.*
