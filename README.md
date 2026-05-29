#  End-to-End AI Framework for Student Performance Analysis, Prediction & Deployment

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org)
[![GitHub](https://img.shields.io/badge/GitHub-meghadeopa-black)](https://github.com/meghadeopa/Student-Performance-AI)

-----------------------------------------------------------------------------------------------------------------------------

# Project Overview

Education systems struggle to identify at-risk students early enough to intervene effectively.
This project builds a complete end-to-end Machine Learning framework that analyses, predicts, and classifies student academic performance and deploys it as a live web application for real-time use.
Instead of waiting for exam results, this system predicts outcomes in advance, giving educators time to act.

----------------------------------------------------------------------------------------------------------------------------

# Problem Statement

Can we predict whether a student will pass or fail and estimate their final score, based on behaviouraland academic input features, before the final exam?

---------------------------------------------------------------------------------------------------------------------------

# Dataset

| Detail     | Info                              |
|------------|-----------------------------------|
| Source     | Kaggle — Student Performance Data |
| Records    | 1,000 students                    |
| Features   | Study hours, attendance,          |
|            | previous scores, parental         |
|            | education, extracurricular        |
| Target     | Final score + Pass/Fail label     |

-----------------------------------------------------------------------------------------------------------------------

# Tech Stack

| Category         | Tools                              |
|------------------|------------------------------------|
| Language         | Python 3.x                         |
| Data Analysis    | Pandas, NumPy                      |
| Visualization    | Matplotlib, Seaborn                |
| Machine Learning | Scikit-learn                       |
| Clustering       | K-Means                            |
| Deployment       | Streamlit                          |
| Version Control  | Git, GitHub                        |

-----------------------------------------------------------------------------------------------------------------------------

# Project Structure

Student-Performance-AI/
│
├── data/
│   └── student_data.csv
│
├── notebooks/
│   ├── 1_EDA_Student.ipynb
│   ├── 2_Model_Training.ipynb
│   ├── 3_My_Addition.ipynb
│   ├── 4_Classification.ipynb
│   └── 5_Advanced_A.ipynb
│
├── app.py              ← Streamlit Web App
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore


--------------------------------------------------------------------------------------------------------------------------

# Approach & Methodology

# Step 1 — Exploratory Data Analysis (EDA)
- Analysed distributions of all features
- Identified and handled missing values and outliers
- Performed correlation analysis
- Key Finding: Study hours and attendance showed
  strongest positive correlation with final scores

# Step 2 — Regression (Predicting Scores)

Compared 9 regression models:

| Model                   | Test R²   |
|-------------------------|-----------|
| ✅ Linear Regression    | **0.8804**|
| Ridge Regression        | 0.8791    |
| Lasso Regression        | 0.8756    |
| Random Forest           | 0.8543    |
| Gradient Boosting       | 0.8621    |
| XGBoost                 | 0.8589    |
| Decision Tree           | 0.8102    |
| SVR                     | 0.8334    |
| KNN Regressor           | 0.8201    |

**Winner: Linear Regression — Test R² = 0.8804**
Model explains 88% of score variance

# Step 3 — Classification (Pass/Fail Prediction)

- Created binary Pass/Fail label (Pass = score ≥ 50)
- Trained classification model on labelled data

| Metric    | Score      |
|-----------|------------|
| Accuracy  | **99.50%** |
| Precision | 99.48%     |
| Recall    | 99.52%     |
| F1 Score  | 99.50%     |


# Step 4 — Clustering (Student Segmentation)

Applied K-Means Clustering (k=3):

| Cluster | Segment           | Characteristics              |
|---------|-------------------|------------------------------|
| 0       |    High Performer | High study hours, attendance |
| 1       |    Average        | Moderate engagement          |
| 2       |    At-Risk        | Low scores, low attendance   |


# Step 5 — Deployment

Deployed as a live Streamlit web application.

Features:
- Input student data via interactive form
- Get instant score prediction
- Get Pass/Fail classification
- See student segment (High/Average/At-Risk)
- Receive personalised recommendations

# Key Results Summary

| Task           | Metric        | Result     |
|----------------|---------------|------------|
| Regression     | Test R²       | 0.8804     |
| Classification | Accuracy      | 99.50%     |
| Clustering     | Segments      | 3 Groups   |
| Deployment     | Platform      | Streamlit  |

# Business Impact

-: Early identification of at-risk students enables timely teacher intervention
-: Personalised recommendations improve student outcomes
-: Scalable to any school or university dataset
-: Reduces dependency on end-of-year results


# How to Run Locally

# Clone the repository
git clone https://github.com/meghadeopa/Student-Performance-AI

# Navigate to folder
cd Student-Performance-AI

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Links

GitHub: [github.com/meghadeopa/Student-Performance-AI](https://github.com/meghadeopa/Student-Performance-AI) 
Live App: Coming Soon 
LinkedIn: [linkedin.com/in/meghadeopa1505](https://linkedin.com/in/meghadeopa1505) 

# Author

**Megha Deopa**
Data Analyst | MBA in AI & ML
megha.deopa13@gmail.com
linkedin.com/in/meghadeopa1505
