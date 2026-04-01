# Detecting Fake Instagram Accounts with Machine Learning

## Project Overview

This project applies **data science and decision science** to detect fake Instagram accounts using machine learning, with a focus on balancing **predictive performance and practical deployment considerations**.

Fake account detection impacts platform trust, safety, and user experience. Rather than optimizing for accuracy alone, this project evaluates models based on their ability to balance:

- fraud detection (sensitivity)  
- protection of real users (specificity)  
- overall model performance (AUC)  

The final model, **Random Forest**, achieved the strongest overall performance:

- **AUC: 0.963**  
- **Sensitivity: 86.96%**  
- **Specificity: 94.20%**  

---

## Problem Statement

Social media platforms must continuously detect fake accounts to maintain **user trust, platform safety, and content quality**. However, identifying fraudulent accounts at scale introduces important trade-offs.

Highly sensitive models can improve fraud detection but risk incorrectly flagging legitimate users, while overly conservative models may fail to detect harmful accounts. Additionally, more complex models may improve accuracy but introduce unnecessary computational overhead.

The challenge is to develop a model that not only performs well statistically, but also balances **detection accuracy, reliability, and practical deployment considerations**.

This project addresses that challenge by evaluating multiple machine learning models and identifying the solution that provides the strongest overall **performance-to-reliability trade-off**.

---

## Dataset

- 696 Instagram accounts  
- Balanced dataset (348 fake / 348 real)  
- No missing values

<p align="center">
  <img src="images/Percent Distribution.png" width="600">
</p>

**Key features:**
- Profile picture  
- Username patterns  
- Followers
- Following
- Bio length  
- External URL
---

## Exploratory Data Analysis

Initial analysis identified key patterns that distinguish fake and real accounts:

- Accounts with a **profile picture** are significantly less likely to be fake  
- Usernames containing **numbers** are strong indicators of fake accounts  
- Fake accounts are less likely to include an **external URL**  
- Engagement-related features (followers, follows, posts) show distinct distributions  

<p align="center">
  <img src="images/Correlation Matrix.png" width="800">
</p>

These patterns highlight that simple, structured features provide strong predictive signals for detecting fraudulent accounts.

---

## Modeling Approach

**Models evaluated:**
- Logistic Regression (baseline + interpretability)  
- K-Nearest Neighbors (benchmark)  
- Random Forest (ensemble model)  

**Validation strategy:**
- 80/20 train-test split  
- 10-fold cross-validation  
- ROC/AUC optimization  

---

## Model Performance

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | AUC |
|------|----------|-----------|-------------|-------------|----------|-----|
| Logistic Regression | 89.86% | **95.08%** | 84.06% | **95.65%** | 0.892 | 0.944 |
| **Random Forest (Best Model)** | **90.58%** | 93.75% | **86.96%** | 94.20% | **0.902** | **0.963** |
| KNN | 83.33% | 85.94% | 79.71% | 86.96% | 0.827 | 0.901 |

**Key Insight:** Random Forest provides the best overall balance of detection performance and reliability.

---

## ROC Curve

<p align="center">
  <img src="images/ROC Curve.png" width="800">
</p>

The Random Forest model achieves the highest AUC, demonstrating strong ability to distinguish between fake and real accounts.

---

## Confusion Matrices 

<p align="center">
  <img src="images/Confusion Matrix.png" width="800">
</p>

**Interpretation:**
- True Positives: correctly identified fake accounts  
- True Negatives: correctly identified real accounts  
- False positives help protect real users  
- False negatives improve platform safety  

---

## Key Insights

- Profile picture strongly indicates real accounts  
- Usernames with numbers increase likelihood of fake accounts  
- Fake accounts exhibit distinct behavioral patterns  

---

## Decision Science Perspective

Model selection was approached as a **trade-off problem**, not just a performance problem.

Instead of asking:  
**Which model is most accurate?**

The analysis focused on:  
**Which model provides the best balance between performance, reliability, and practical deployment?**

Random Forest delivers the strongest overall performance while remaining computationally practical, making it the **decision-optimal model**.

---

## Final Recommendation

**Random Forest** is selected as the optimal model due to:

- Highest overall predictive performance  
- Strong balance between sensitivity and specificity  
- Robust and scalable modeling approach  

---

## Tools

- R (`caret`, `randomForest`, `pROC`, `ggplot2`, `dplyr`, `tidyr`)

---

## Key Skills Demonstrated

- Machine learning (classification)  
- Model evaluation (AUC, ROC, F1-score)  
- Feature importance analysis  
- Exploratory data analysis  
- Decision science / trade-off analysis  
