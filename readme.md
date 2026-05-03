# 🧠 Hybrid Deepfake Detection System

A **Hybrid Deep Learning Framework** for detecting deepfake images using a combination of:

* 🌍 **Global Image Model** (MobileNetV2)
* 🧑 **Face-Focused Model** (EfficientNetB0)
* 🔍 **Grad-CAM Visualization** for explainability
* 🌐 **Django Web App** for real-time inference

---

## 🚀 Overview

Deepfakes primarily manipulate **facial regions**, but not all images contain clear faces.
To address this, this project uses a **dual-model pipeline**:

* If a face is detected → use **EfficientNet (Face Model)**
* Otherwise → use **MobileNetV2 (Global Model)**

This hybrid approach improves **accuracy, robustness, and real-world usability**.

---

## 🏗️ System Architecture

```
Input Image
     ↓
Preprocessing
     ↓
Face Detection
   /        \
Face       No Face
 ↓            ↓
EfficientNet   MobileNetV2
 ↓            ↓
Prediction + Confidence
     ↓
Grad-CAM Visualization
     ↓
Final Output (Real / Fake)
```

---

## ✨ Features

* 🔄 **Hybrid Model Selection**
* ⚡ **Fast Inference (MobileNetV2)**
* 🎯 **High Accuracy on Faces (EfficientNet)**
* 🔍 **Explainability with Grad-CAM**
* 🌐 **Django-based Web Interface**
* 🧩 Modular and scalable architecture

---

## 📁 Project Structure

```
Project v2/
│
├── core/
│   ├── load_models.py
│   ├── gradcam.py
│   ├── hybrid_pipeline.py
│   ├── face_detector.py
│
├── detector/                # Django app
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│
├── saved_models/
│   ├── mobilenetv2_global_best.keras
│   ├── face_model_final.keras
│
├── static/
├── manage.py
```

---

## 🧠 Models Used

### 1. Global Model

* **Architecture:** MobileNetV2
* **Purpose:** Detect manipulation across entire image
* **Strength:** Fast, lightweight

---

### 2. Face Model

* **Architecture:** EfficientNetB0
* **Purpose:** Detect subtle facial artifacts
* **Strength:** High accuracy on face-based deepfakes

---

## 🔍 Grad-CAM Visualization

The system provides visual explanations by highlighting:

* Manipulated facial regions
* Background inconsistencies
* Model attention areas

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python manage.py runserver
```

Open in browser:

```
http://127.0.0.1:8000/
```

---

## 🧪 Usage

1. Upload an image
2. System detects if a face is present
3. Selects appropriate model:

   * Face → EfficientNet
   * No face → MobileNetV2
4. Returns:

   * Prediction (Real / Fake)
   * Confidence score
   * Grad-CAM visualization

---

## 📊 Performance Benefits

| Feature                 | Single Model | Hybrid Model |
| ----------------------- | ------------ | ------------ |
| Face Deepfake Detection | Medium       | High         |
| General Image Detection | Medium       | High         |
| Explainability          | Yes          | Yes          |
| Robustness              | Moderate     | High         |

---

## 🧠 Research Contribution

* Hybrid CNN architecture improves detection reliability
* Model specialization reduces false positives
* Supports real-world mixed-image scenarios
* Provides explainable AI via Grad-CAM

---

## 🔮 Future Improvements

* 🎥 Video deepfake detection
* 👥 Multi-face detection support
* ⚡ Real-time webcam inference
* 🌐 REST API for frontend integration
* 📱 Mobile deployment

---

## 👨‍💻 Tech Stack

* **Backend:** Django
* **ML Framework:** TensorFlow / Keras
* **Models:** MobileNetV2, EfficientNetB0
* **Visualization:** Grad-CAM
* **Training:** Google Colab GPU

---

## 📌 Notes

* Ensure models are placed in `saved_models/`
* Use correct preprocessing:

  * MobileNet → global model
  * EfficientNet → face model

---

## 📄 License

This project is for **educational and research purposes**.

---

## 🙌 Acknowledgements

* TensorFlow & Keras
* Open-source deepfake datasets
* Google Colab

---
