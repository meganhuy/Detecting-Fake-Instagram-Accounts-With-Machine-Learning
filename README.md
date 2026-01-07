# Detecting_Instagram_Accounts_Project

## Project Overview

This project focuses on detecting fake and real Instagram accounts efficiently and cost-effectively. Our goal is to build several machine learning models and select the model that maintains and improves detection accuracy while reducing annual cost.

## Objectives

-Develop an ML model to classify Instagram accounts as fake or real.
-Optimize for accuracy, sensitivity, and specificity while reducing cost.
-Deploy the model for scalability and cost efficiency.

## Dataset

Source: Kaggle – Instagram Fake and Real Accounts Dataset

## Key Findings:

-Profile Picture: Strong negative correlation with fake accounts (r = -0.62).
-Username with Numbers: Moderate positive correlation with fake accounts (r = 0.57).
-Fake accounts are less likely to have profile pictures or external URLs.
-Fake accounts are more likely to have usernames resembling full names or containing numbers.

## Models Tested

1.Logistic Regression (Baseline model)
2.K-Nearest Neighbors (KNN)	(Most cost-effective)
3.Random Forest	(Best Performing)

## Metric:	AUC,	Sensitivity,	Specificity	

### The Random Forest model demonstrated superior predictive performance and robustness.

##  Cost Comparison
Model	Deployment Cost	Net Cost	Annual Estimate
Current Model	—	—	$240,000
Proposed Model —	—		$60,000 Total

### Company saves approximately $180,000 per year.

 ## Technical Workflow

1. Data Cleaning & Preprocessing
2. Handle missing values, feature encoding, and scaling.
3. Exploratory Data Analysis (EDA)
4. Correlation, feature visualization, and statistical insights.
5.  Model Training (Random Forest, Logistic Regression, and KNN models)

##  Results & Conclusion

Best Model: Random Forest
Performance: AUC = 0.96
Savings: $180,000 annually
Outcome: Increases users’ trust and safety while maintaining platform integrity at lower cost.

###  Random Forest offers the best balance of accuracy and cost-efficiency.
