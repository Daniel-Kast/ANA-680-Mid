# Diabetes Risk Prediction App (ANA 500 / ANA 680 Midterm Project)

This project predicts diabetes risk among U.S. adults using a neural network (MLP) trained on the **Behavioral Risk Factor Surveillance System (BRFSS)** dataset.  
The model is deployed as a live web application on Heroku:

🔗 **Live App:** https://ana680mid-b4b873caaac4.herokuapp.com/

---

## 📌 Project Overview

Diabetes is a major public health concern in the United States.  
Using BRFSS survey data, this project explores **demographic, behavioral, and health‑related factors** associated with diabetes diagnosis.

The final deployed model is a **Multi‑Layer Perceptron (MLP)** trained on the **top 4 predictive features**, selected using mutual information.  
The app allows users to enter these four variables and receive:

- A binary prediction (Diabetic / Not Diabetic)
- A probability score (0–1)

---

## 🧠 Machine Learning Workflow

### **1. Data Preparation**
- Combined categorical, numeric, and ordinal BRFSS files  
- Cleaned missing values and imputed where appropriate  
- Removed duplicate diabetes indicators  
- Created a binary target:  
  - `1` = Diabetic  
  - `0` = Not diabetic  

### **2. Feature Selection**
Used **SelectKBest + Mutual Information** to identify the **top 4 predictors**.

### **3. Model Training**
- StandardScaler for normalization  
- MLPClassifier with:
  - Hidden layers: (32, 16)
  - Activation: ReLU
  - Solver: Adam
  - Max iterations: 500

### **4. Deployment**
- Flask web app  
- HTML form for user input  
- Hosted on Heroku  
- Model + scaler + feature list saved as `.pkl` files  

---

## 🧩 App Input Key (BRFSS Coding Guide)

Your app uses four BRFSS variables.  
Below are the **valid ranges and meanings** for each one.

---

### **1. EMPLOY1 — Employment Status**
| Code | Meaning |
|------|---------|
| 1 | Employed for wages |
| 2 | Self‑employed |
| 3 | Out of work (≥ 1 year) |
| 4 | Out of work (< 1 year) |
| 5 | Homemaker |
| 6 | Student |
| 7 | Retired |
| 8 | Unable to work |

---

### **2. GENHLTH — General Health Rating**
| Code | Meaning |
|------|---------|
| 1 | Excellent |
| 2 | Very good |
| 3 | Good |
| 4 | Fair |
| 5 | Poor |

---

### **3. _AGEG5YR — Age Group (5‑year bins)**
| Code | Age Range |
|------|-----------|
| 1 | 18–24 |
| 2 | 25–29 |
| 3 | 30–34 |
| 4 | 35–39 |
| 5 | 40–44 |
| 6 | 45–49 |
| 7 | 50–54 |
| 8 | 55–59 |
| 9 | 60–64 |
| 10 | 65–69 |
| 11 | 70–74 |
| 12 | 75–79 |
| 13 | 80+ |

---

### **4. BMI5CAT — BMI Category**
| Code | Meaning |
|------|---------|
| 1 | Underweight |
| 2 | Normal weight |
| 3 | Overweight |
| 4 | Obese |

---

## 🗂 Repository Structure

/project
│
├── app.py
├── mlp_model.pkl
├── scaler.pkl
├── selected_features.pkl
├── combined_clean.csv
├── requirements.txt
├── Procfile
└── templates/
    └── index.html

---

## 🚀 How to Run Locally

pip install -r requirements.txt
python app.py

## Then open
http://127.0.0.1:5000

👨‍💻 Author
Daniel Kast
ANA 500 / ANA 680
National University


