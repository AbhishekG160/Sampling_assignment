# Credit Card Fraud Detection using Sampling Techniques and Machine Learning

## 📌 Project Overview

This project focuses on **credit card fraud detection** using machine learning techniques on an **imbalanced dataset**.
Since fraud cases are rare compared to normal transactions, the notebook explores **sampling strategies** and evaluates how different **classification models** perform under these techniques.

The main goal is to **identify the best combination of sampling method and classifier** that yields the highest prediction accuracy.

---

## 🎯 Objectives

* Handle severe class imbalance in credit card transaction data
* Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset
* Implement and compare **five different sampling techniques**
* Evaluate **five machine learning classifiers** across all sampling methods
* Identify the **best sampling technique for each model**

---



## 📊 Dataset Information

* **Source:** Credit Card Transaction Dataset
* **Total Records:** 772 transactions
* **Features:**

  * 30 numerical features
  * V1–V28 (anonymized PCA features)
  * `Time`, `Amount`
* **Target Variable:**

  * `Class = 0` → Legitimate transaction
  * `Class = 1` → Fraudulent transaction

---

## ⚖️ Key Challenge -> Class Distribution

### Original Dataset (Highly Imbalanced)

* Legitimate (Class 0): **763 transactions (98.8%)**
* Fraudulent (Class 1): **9 transactions (1.2%)**

This extreme imbalance makes direct model training unreliable, as models tend to predict only the majority class.

### After SMOTE Oversampling (Balanced Dataset)

* Class 0: **763 transactions (50%)**
* Class 1: **763 transactions (50%)**
* **Total:** 1,526 transactions

SMOTE generates **synthetic fraud samples** to ensure equal class representation without duplicating existing records.


---
## 🔄 Workflow Summary

---

### 1️⃣ Data Inspection and Preprocessing

#### 📊 Class Distribution Analysis (Illustration)

To visualize the imbalance, a **count plot** is created for the `Class` column.

**Illustration Description:**
A bar graph clearly shows:

* A **very tall bar** for Class `0` (non-fraud)
* A **very short bar** for Class `1` (fraud)

This confirms that the dataset is **severely imbalanced**, justifying the need for sampling techniques.

**Steps performed:**

* Loaded the credit card transaction dataset
* Separated features (`X`) and target variable (`y`)
* Applied **SMOTE** to the training data only
* Created a **fully balanced dataset** for fair model learning

---

### 2️⃣ Sampling Techniques Implemented

| Sampling Method     | Description                  | Implementation                             |
| ------------------- | ---------------------------- | ------------------------------------------ |
| Random Sampling     | Random selection of records  | 80% random selection without replacement   |
| Stratified Sampling | Preserves class distribution | Maintains 50:50 class ratio                |
| Bootstrap Sampling  | Sampling with replacement    | Same size as original dataset              |
| Systematic Sampling | Selects every k-th record    | k = 2 (every 2nd row)                      |
| Cluster Sampling    | Samples grouped clusters     | Data divided into 3 clusters, one selected |

Each sampling method generates a **new training subset** from the balanced dataset.

---

### 3️⃣ Machine Learning Models

The following five classifiers were evaluated:

* **Logistic Regression** (`max_iter = 1000`)
* **Decision Tree Classifier**
* **Random Forest Classifier** (`n_estimators = 50`)
* **Support Vector Machine (SVM)** with RBF kernel
* **Gaussian Naive Bayes**

📌 **Why Sampling?**
To prevent models from favoring the majority class and to improve fraud detection capability.

Every model is trained on **sampled training data** and tested on the **same test set** for fairness.

---

### 4️⃣ Experimental Design

* Created **5 independent random samples** from the balanced dataset (80% each)
* Applied **all sampling techniques** to each sample
* Trained **all five models** on every sampled dataset
* Used a **70–30 train–test split**
* Total experiments conducted:
  **5 samples × 5 sampling methods × 5 models = 125 experiments**

This design ensures **robustness and reduced randomness bias**.

---

## 📈 Results Summary

### 🏆 Top Performing Configurations

| Rank | Configuration                                  | Accuracy   |
| ---- | ---------------------------------------------- | ---------- |
| 🥇 1 | Sample 1 · Stratified Sampling · Random Forest | **99.73%** |
| 🥈 2 | Sample 2 · Random Sampling · Random Forest     | 99.46%     |
| 🥈 2 | Sample 4 · Stratified Sampling · Random Forest | 99.46%     |
| 🥈 2 | Sample 4 · Bootstrap Sampling · Random Forest  | 99.46%     |
| 🥉 3 | Sample 3 · Bootstrap Sampling · Random Forest  | 99.18%     |

---

### ✅ Best Sampling Technique per Model

| Model               | Best Sampling Method | Accuracy     |
| ------------------- | -------------------- | ------------ |
| Logistic Regression | Bootstrap Sampling   | 94.55%       |
| Decision Tree       | Random Sampling      | 98.64%       |
| Random Forest       | Stratified Sampling  | **99.73%** ✨ |
| SVM                 | Bootstrap Sampling   | 73.02%       |
| Naive Bayes         | Stratified Sampling  | 88.56%       |

---

### 7️⃣ Model Evaluation

For each combination of:

* Sampling method
* Machine learning model

The following metrics are computed:

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report** (Precision, Recall, F1-score)

**Illustration Description:**
Confusion matrices indicate:

* True Positives (fraud correctly detected)
* False Positives (normal transactions flagged as fraud)
* False Negatives (missed fraud cases)

---

### 8️⃣ Results Comparison

* All results are stored in a dictionary
* Converted into a **sorted DataFrame**
* Top-performing combinations are displayed

📌 **Best Sampling Selection**
For each model, the notebook identifies:

* The **sampling technique** that achieved the **highest accuracy**

This makes it easy to compare which sampling method works best for each classifier.

---

## 🔍 Key Findings

### 1️⃣ Random Forest Dominates

* Achieved the highest accuracy across all sampling techniques
* Best performance: **99.73%**
* Highly robust to sampling variations

### 2️⃣ Stratified Sampling is Most Reliable

* Preserves class distribution
* Best suited for Random Forest and Naive Bayes
* Ensures representative training data

### 3️⃣ Bootstrap Sampling Helps Linear Models

* Improves generalization for Logistic Regression
* Slight improvement for SVM, though overall performance remains low

### 4️⃣ SVM Underperforms

* Accuracy ranges between **59%–73%**
* Requires extensive tuning
* Not suitable for this dataset in its current form

### 5️⃣ Sampling Impact Depends on Model Type

* Tree-based models: Less sensitive to sampling choice
* Probabilistic and linear models: Strongly affected by sampling
* SVM: Poor performance regardless of technique

---

## 🧠 Learning Takeaways

* Imbalanced datasets can severely mislead ML models
* Sampling is essential in fraud detection problems
* No single model is best — performance depends on preprocessing choices
* Evaluation must go beyond accuracy and include confusion matrices and reports

---

## 📌 Conclusion

This project demonstrates how **sampling techniques combined with machine learning** can significantly improve fraud detection performance.
It provides a practical framework for tackling **real-world imbalanced classification problems**.

---

**Author:** Abhishek Gupta (**102316027**)

**Domain:** Machine Learning | Data Science | Fraud Detection
