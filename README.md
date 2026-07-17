# 🧠 Brain Tumor Detection & Diagnostics Web Application

An end-to-end, clinically-oriented computer-aided diagnostics (CAD) system leveraging **Deep Learning (Convolutional Neural Networks)** to classify brain MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary tumor, or No Tumor (Healthy Brain)**. 

The project includes an interactive, production-ready, and highly user-friendly **Streamlit** dashboard designed with dual-language capabilities (Arabic & English), integrated account simulation, and a direct developer support channel.

---

## 🚀 Key Features

* **Custom CNN Architecture:** Handcrafted Convolutional Neural Network built using **TensorFlow/Keras**, optimized with Dropout regularization ($0.5$) to prevent overfitting, achieving a robust **73.35% accuracy** on unseen test datasets.
* **Interactive Dual-Language UI (Ar/En):** Seamless on-the-fly toggling between Arabic and English, allowing patients and medical professionals to read the diagnostic reports in their preferred language.
* **Medically-Contextualized Reports:** For every prediction, the application instantly generates detailed medical information:
  * Comprehensive condition descriptions.
  * Common associated symptoms.
  * Recommended clinical next steps and actionable advice.
* **Interactive Control Sidebar:**
  * **Account Simulation:** Simulated login options via Email or Facebook.
  * **Settings Panel:** Interactive threshold adjustments (50%–95%) and Dark Mode toggle.
  * **Integrated Support Mailer:** A direct feedback tool that automatically structures and drafts support emails to the developer (`yosefelosely@gmail.com`).
* **Clean, Modern UI/UX:** Styled with responsive, custom CSS, professional color palettes (alert red for anomalies, success green for healthy scans), and fixed designer branding.

---

## 📁 Project Structure

```bash
Brain_Tumor_Project/
├── app.py                  # Streamlit Web Application (UI, Logic, & Routing)
├── brain_tumor_model.h5    # Trained & Optimized Custom CNN Model
├── main.ipynb              # Jupyter Notebook (Data preprocessing, model architecture, training & evaluation)
├── requirements.txt        # Package dependencies for easy deployment
└── README.md               # Repository documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Create a `requirements.txt` file and install:
```bash
pip install -r requirements.txt
```
*Note: Make sure to include `streamlit`, `tensorflow`, `pillow`, and `numpy` in your environment.*

### 4. Run the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501` to use the application!

---

## 📊 Model Training Summary

The custom deep learning model was trained end-to-end:
* **Preprocessing:** Input MRI slices resized to $150 \times 150 \times 3$, normalized to $[0, 1]$, and shuffled.
* **Training Epochs:** Optimized over **30 epochs** using the Adam optimizer and Categorical Cross-Entropy loss.
* **Regularization:** Heavy dropout layer ($50\%$) utilized prior to the classification head to ensure high generalization performance.
* **Performance:** Reached **73.35% Generalization Accuracy** on the testing dataset with high confidence thresholds.

---

## 👨‍💻 Developer & Designer

Developed with ❤️ by **Youssef Elosely**  
*Medical Biophysics | Faculty of Science*  
*📧 Support Contact: [yosefelosely@gmail.com](mailto:yosefelosely@gmail.com)*

---
*Disclaimer: This application is a computer-aided diagnostic prototype designed for educational and research purposes. It should not be used as a standalone replacement for professional medical consultation.*
