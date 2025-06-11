

## 🏥 **Postoperative Pulmonary Complications Prediction**

### 🚀 Project Overview

This is a **machine learning model** for **Postoperative Pulmonary Complications (PPCs) prediction**, utilizing **Logistic Regression** and **CatBoost** models. It predicts the likelihood of postoperative pulmonary complications based on a patient's basic information. The app includes:

* Data preprocessing and model training
* Risk prediction with SHAP explainability
* Risk classification based on the optimal threshold value

---

### ⚙️ **Installation and Deployment**

#### 1. **Clone the Project**

Start by cloning this repository:

```bash
git clone https://github.com/harkool/ppcs-web.git
cd ppcs-web
```

#### 2. **Install Dependencies**

Ensure you have Python 3.x installed, then run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

> Required Python libraries include: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `catboost`, `shap`, `matplotlib`.

#### 3. **Install System Dependencies (Streamlit Cloud Deployment)**

Create a `packages.txt` file in the project root with the following content to install system dependencies (e.g., `libgomp1`):

```txt
libgomp1
libstdc++6
libgl1-mesa-glx
```

These dependencies are essential for the **CatBoost** model, particularly for OpenMP and C++ compilation support.

#### 4. **Upload to GitHub and Deploy to Streamlit Cloud**

1. Create a new repository on GitHub.
2. Upload and push the project files (`git push`).
3. Deploy the app via [Streamlit Cloud](https://share.streamlit.io).

#### 5. **Running the App Locally**

To run the app locally, use:

```bash
streamlit run app.py
```

### 🔑 **Configuration Files**

* `requirements.txt`: Contains all Python package dependencies.
* `packages.txt`: Contains system-level library dependencies that will be installed by Streamlit Cloud.

---

### 🔧 **Features**

1. **Model Selection**:

   * **Preoperative only**: Prediction based on preoperative data only.
   * **Preoperative and Intraoperative**: Prediction based on both preoperative and intraoperative data.

2. **Input Data**:
   Users need to input the following information in the sidebar:

   * Age
   * CRP (C-Reactive Protein)
   * Diabetes history
   * Smoking
   * Preoperative anemia
   * Preoperative respiratory tract infection
   * Preoperative SpO2
   * ASA Physical Status Classification
   * Intraoperative blood loss (for the "Preoperative and Intraoperative" model)

3. **Prediction and Risk Classification**:

   * Patients are classified as **High Risk** or **Low Risk** based on the **best classification threshold**.
   * Displays prediction probability and SHAP explanation plots.

---

### 📊 **Model Evaluation**

1. **Logistic Regression**:

   * Uses `class_weight="balanced"` to address class imbalance.
   * Evaluation Metrics: Accuracy, F1 Score, ROC AUC.

2. **CatBoost**:

   * Uses `scale_pos_weight` to handle class imbalance.
   * Evaluation Metrics: Accuracy, F1 Score, ROC AUC.

---

### 📝 **Notes**

* When deploying to **Streamlit Cloud**, ensure you use **`packages.txt`** to install system dependencies.
* If you're using GitHub, follow the steps to deploy your app to Streamlit Cloud.

---

### 📬 **Contact Information**

* **Developer**: \[Hao Liu]
* **Email**: \[lenhartkoo@foxmail.com]
* **GitHub**: [https://github.com/harkool](https://github.com/harkool)

