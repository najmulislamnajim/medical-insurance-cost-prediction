
# Medical Insurance Cost Prediction (End-to-End ML + Gradio)

Predict medical insurance charges based on user demographics and lifestyle information using a complete Machine Learning workflow: preprocessing → pipeline → cross-validation → hyperparameter tuning → evaluation → Gradio web app.

---

## 📌 Project Overview
This project builds a regression model to predict **medical insurance charges (`charges`)** from the following inputs:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

**Goal:** Create a clean, reproducible ML pipeline and a simple web interface for real-time predictions.

---

## 📂 Dataset
Kaggle: **Medical Insurance Cost Prediction**  
https://www.kaggle.com/datasets/mirichoi0218/insurance

---

## ✅ Key Features
- Data preprocessing 
- End-to-end **Pipeline** (preprocessing + model combined)
- **Cross-Validation** for robust scoring
- **Hyperparameter Tuning** (GridSearchCV / RandomizedSearchCV)
- Final evaluation on test set (R² / MAE / RMSE)
- **Gradio** web interface for user-friendly predictions
- Model saving (`model.pkl`) for easy reuse

---

## 🧹 Preprocessing Steps
This project includes at least 5 distinct preprocessing steps:
1. **Missing value check/handling** (median/mode strategy if needed)
2. **Duplicate removal**
3. **Outlier detection** (boxplot + IQR-based inspection)
4. **Categorical encoding** (OneHotEncoder)
5. **Scaling** numeric features (StandardScaler)

> Note: The model is trained using a pipeline to prevent data leakage.

---

## 🧠 Model
Primary model: **RandomForestRegressor**  
**Why this model?**
- Works well on tabular datasets
- Captures non-linear relationships
- Strong baseline performance with minimal assumptions

---

## 📏 Evaluation Metrics
- **R² Score** (higher is better)
- **MAE** (lower is better)
- **RMSE** (lower is better)

---

## 🧪 Cross-Validation & Hyperparameter Tuning
- Cross-validation performed using `cv=5`
- Hyperparameter search using `GridSearchCV` (or `RandomizedSearchCV` for faster runs)
- Best model selected based on the chosen scoring metric

---

## 🌐 Gradio Web App
A Gradio interface is included to:
- take user inputs (`age`, `bmi`, `children`, `sex`, `smoker`, `region`)
- show predicted insurance cost instantly

---

## 🧰 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Gradio

---

## 📁 Project Structure (Suggested)
```

.
├── app.py                 # Gradio web app
├── train_model.py         # training + tuning + save model.pkl
├── model.pkl              # saved best pipeline model
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
````

---

## ▶️ Run Training (Generate model.pkl)

```bash
python train_model.py
```

This will train the model and save the best pipeline as `model.pkl`.

---

## 🚀 Run Gradio App

```bash
python app.py
```

Then open the local URL shown in the terminal.

---

## 👤 Author

**Najmul Islam**

```
najmulislamru@gmail.com
```

---

## 📜 License

This project is for educational purposes.

---

## 🙏 Acknowledgements

* Dataset: Kaggle / mirichoi0218
* Tools: Scikit-learn, Gradio

