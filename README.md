# Facial Emotion Detection

A deep learning project that detects human emotions from facial expressions using the **Face Expression Recognition Dataset** (Jonathan O’Heix) from Kaggle.  
The goal is to classify faces into emotion classes and support both batch inference and real-time webcam detection.

---

## 📌 Project Overview
- **Dataset:** [Face Expression Recognition Dataset by Jonathan O’Heix (Kaggle)](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)  
- **Task:** Multi-class classification (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)  
- **Techniques:** Image preprocessing, Data Augmentation, Convolutional Neural Networks (or transfer learning), Real-time face detection with OpenCV

---

## ⚙️ Workflow
1. **Exploratory Data Analysis (EDA)**  
   - Check class balance, image shapes, and label distributions.  
   - Visualize example faces per emotion and class frequencies.

2. **Data Cleaning & Preprocessing**  
   - Convert images to grayscale (if needed) and resize (e.g., 48×48 or 64×64).  
   - Normalize pixel values and apply augmentation (rotation, shifts, flips).  
   - Create stratified train/validation/test splits.

3. **Modeling**  
   - Baseline: custom small CNN.  
   - Experiments: transfer learning with MobileNet/EfficientNet or deeper CNNs.  
   - Use callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

4. **Evaluation & Inference**  
   - Report per-class accuracy, confusion matrix, precision/recall/F1.  
   - Save best model to `models/` for inference.

---

## 🛠️ Requirements
    python 3.10
    tensorflow
    opencv-python
    numpy
    pandas
    matplotlib
    scikit-learn
    tqdm
    mtcnn
    keras

Install dependencies:
```bash
pip install -r requirements.txt
```
