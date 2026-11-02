# Brain Tumor MRI Classification & Explainable AI (Grad-CAM)

In this project, a pre-trained **ResNet34** convolutional neural network is fine-tuned in PyTorch to classify human brain MRI scans into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. 

Four training approaches—exploring learning rate tuning, model checkpointing, partial layer freezing, and data augmentation—are compared to improve generalization on unseen test scans. Finally, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is implemented from scratch to visualize which regions of the MRI scan the model focuses on when making both correct and incorrect predictions.

> **Disclaimer:** This project is an educational deep learning and explainability study on public, anonymized MRI data and is not intended for clinical diagnosis.

---

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar on Kaggle, containing **7,200** brain MRI scans balanced across 4 classes:

| Class | Training + Validation (`80/20` split) | Test Set |
| :--- | :---: | :---: |
| **Glioma** | 1,400 | 400 |
| **Meningioma** | 1,400 | 400 |
| **Pituitary Tumor** | 1,400 | 400 |
| **No Tumor** | 1,400 | 400 |
| **Total** | **5,600** | **1,600** |

All images are resized to `224 × 224` and converted to 3-channel normalized tensors (`[0, 1]`) to match the input format of `ResNet34`.

---

## Model & Training Approaches

A pre-trained `ResNet34` (`ResNet34_Weights.DEFAULT`) was adapted for 4-class classification by replacing the final fully connected layer (`model.fc = nn.Linear(512, 4)`), trained using `CrossEntropyLoss` and the `Adam` optimizer.

| Approach | Configuration | Final Train Acc | Final Val Acc | **Test Accuracy** | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Approach 1** | All layers unfrozen, `lr=0.001`, no checkpointing | 97.46% | 92.86% | **88.50%** | Step size (`lr=0.001`) was too large for pre-trained weights, causing unstable validation loss and overfitting. |
| **Approach 2** | All layers unfrozen, `lr=0.0001` + **Model Checkpointing** | 99.69% | 97.86% | **94.19%** | Lowering the learning rate and saving the best validation checkpoint stabilized training. |
| **Approach 3** | Frozen `conv1`–`layer3`, unfrozen `layer4` + `fc`, `lr=0.0001` | 99.71% | 97.77% | **94.31%** | Preserved early edge/texture detectors while adapting high-level features in `layer4`. |
| **Approach 4** | Unfrozen `layer3`, `layer4` + `fc`, `lr=0.0001` + `RandomHorizontalFlip()` | 99.26% | 99.20% | **95.12%** | **Best model.** Horizontal flips match the brain's left-right hemisphere symmetry without introducing rotation artifacts. |

---

## Evaluation & Analysis (Best Model — Approach 4)

### 1. Classification Report (1,600 Test Scans)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **glioma** | 0.99 | 0.83 | 0.90 | 400 |
| **meningioma** | 0.93 | 0.98 | 0.95 | 400 |
| **notumor** | 0.91 | 1.00 | 0.95 | 400 |
| **pituitary** | 0.99 | 0.99 | 0.99 | 400 |
| **Macro / Weighted Avg** | **0.95** | **0.95** | **0.95** | **1,600** |

* **Strengths:** `pituitary` achieved a `0.99` F1-score, and `notumor` reached `1.00` recall (zero healthy brains were misclassified as having a tumor).
* **Main Weakness:** `glioma` had `0.99` precision but `0.83` recall—some gliomas were misclassified as `meningioma` (40 cases) or missed as `notumor` (27 false negatives), which is the most critical error to minimize in a clinical setting.

### 2. Grad-CAM Explainability
Grad-CAM was implemented from scratch on the 512 feature maps (`7 × 7`) of `model.layer4`:
* **Correct Predictions:** Heatmaps highlight the actual tumor mass (or broad healthy brain tissue for `notumor`), confirming the model learned clinically relevant features rather than background artifacts.
* **Predicted Tumor, Wrong Type:** Confusions occur when a tumor is located near another class's typical anatomical region (e.g., a low-sitting `meningioma` near the pituitary gland).
* **Missed Tumors (False Negatives):** In missed tumor scans, heatmaps reveal that the model was distracted by bright skull borders or sinus cavities instead of focusing on fainter tumor tissue.

---

## Project Structure

* `brain-tumor-detection.ipynb` — Complete notebook containing:
  * **Step 1:** Load Data (via `kagglehub`)
  * **Step 2:** Exploratory Data Analysis (EDA)
  * **Step 3:** Preprocessing & `DataLoader` Setup
  * **Step 4:** Model Setup (`ResNet34` Transfer Learning)
  * **Step 5:** Training & Validation Loop (with Checkpointing & Augmentation)
  * **Step 6:** Evaluation (Test Accuracy, Confusion Matrix, Precision/Recall/F1)
  * **Step 7:** Visualization (Grad-CAM Heatmaps) & Analysis
* `requirements.txt` — Python dependencies

---

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run `brain-tumor-detection.ipynb` in Jupyter Notebook or VS Code:
   ```bash
   jupyter notebook brain-tumor-detection.ipynb
   ```
   *(The dataset downloads automatically in Step 1 via `kagglehub`.)*

---

## References

1. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. IEEE International Conference on Computer Vision (ICCV). [arXiv:1610.02391](https://arxiv.org/pdf/1610.02391) | [GitHub: ramprs/grad-cam](https://github.com/ramprs/grad-cam/)
2. Nickparvar, M. (2021). *Brain Tumor MRI Dataset*. Kaggle. [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

