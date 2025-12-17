# Assingment-5

# 🧠 Brain Tumor Classification using Deep Learning

An AI-powered web application for brain tumor classification from MRI images using EfficientNet and Grad-CAM, deployed with Streamlit.

---

## 🚀 Features
- Upload brain MRI images
- Predict tumor type (Glioma, Meningioma, Pituitary, No Tumor)
- Display model confidence scores
- Grad-CAM visualization for explainability
- Ethical, academic & medical disclaimers

---

## 🧠 Model Architecture
- EfficientNetB0 (pretrained on ImageNet)
- Fine-tuned on brain MRI dataset
- Softmax confidence output

---

## 🖥️ Web Application
Built using **Streamlit**, allowing:
- Real-time inference
- Interactive visualization
- User-friendly UI

---

## 🔍 Explainable AI
Grad-CAM is used to highlight regions influencing the model’s prediction.

---

## ⚖️ Disclaimer
This project is for **educational and research purposes only**.  
It is **not a medical device** and must not be used for clinical diagnosis.

---

## 📚 Citation
Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
streamlit run app.py
