# 🦷 Automated Oral Disease Classification using Deep Learning

## 📌 Overview
This project focuses on building an AI-based system to automatically detect and classify oral diseases from dental images using deep learning techniques. The goal is to assist in early diagnosis and improve healthcare accessibility.

---

## 🎯 Objective
- To classify oral diseases from images into 6 categories
- To improve diagnostic accuracy using deep learning
- To compare multiple CNN architectures for performance

---

## 📊 Dataset
- Total Images: 12,320+
- Classes:
  - Calculus
  - Dental Caries
  - Gingivitis
  - Mouth Ulcer
  - Tooth Discoloration
  - Hypodontia

- Data Split:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

---

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn

---

## 🧠 Models Used
- EfficientNet-B4
- EfficientNet-B5
- DenseNet-169 (Best Performing Model)

---

## 🔄 Methodology

### 1. Preprocessing
- Image resizing (B4: 380×380, B5: 456×456)
- Normalization (0–1 scaling)

### 2. Data Augmentation
- Horizontal & vertical flipping
- Brightness & contrast adjustment
- Saturation variation

### 3. Transfer Learning
- Used pretrained models (ImageNet)
- Replaced top layers with custom classifier

### 4. Fine-Tuning
- Step 1: Freeze pretrained layers
- Step 2: Train classifier
- Step 3: Gradually unfreeze layers

### 5. Handling Class Imbalance
- Class-weighted loss
- Focal loss

---

## 📈 Results

| Model              | Accuracy | Weighted F1-score |
|-------------------|---------|------------------|
| EfficientNet-B4   | 86.42%  | 86.61%           |
| EfficientNet-B5   | 90.37%  | 90.44%           |
| DenseNet-169      | **95.29%** | **93.28%**      |

---

## 🔍 Key Features
- Multi-class classification (6 classes)
- High accuracy with DenseNet-169
- Robust training using augmentation and fine-tuning
- Handles class imbalance effectively

---

## 🧾 Output
- Input: Oral cavity image
- Output: Predicted disease class with probability scores

---

## 🚀 Future Improvements
- Real-time detection system
- Mobile/web application deployment
- Integration with clinical systems
- Explainable AI (Grad-CAM)

---

## 📌 Conclusion
DenseNet-169 outperformed other models due to its dense connectivity and feature reuse capability, making it effective for detecting subtle differences in oral diseases.

---

## 👨‍💻 Author
Shiva Chandra
