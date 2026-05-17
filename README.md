# IBM HR Attrition Prediction

A binary classification project to predict whether an employee is likely to leave the company, using the IBM HR Analytics Employee Attrition dataset.

## Results

| Metric | Default Threshold (0.5) | Tuned Threshold (0.3) |
|---|---|---|
| Accuracy | 80% | 63% |
| F1 (Attrition) | 0.48 | 0.42 |
| Recall (Attrition) | 0.57 | 0.85 |
| Precision (Attrition) | 0.41 | 0.28 |

Threshold was tuned to prioritize recall. Catching 40 out of 47 employees who actually left is more valuable for HR than maintaining high accuracy on a heavily imbalanced dataset.

## Dataset

- **Source:** [IBM HR Analytics on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 rows, 35 features
- **Target:** `Attrition`, `Yes` (left) or `No` (stayed)
- **Class split:** 84% stayed, 16% left

## Project Structure

```
ibm-hr-attrition/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/
│   ├── xgboost_ibm_attrition.pkl
│   ├── ohe_encoder.pkl
│   └── threshold.pkl
├── notebook/
│   └── notebook.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Pipeline

**1. Data Cleaning**

No missing values were found across all 35 columns. Three constant columns were identified and dropped: `EmployeeCount`, `StandardHours`, and `Over18`. These had zero variance across all 1,470 rows and contributed no predictive information. `EmployeeNumber` was also dropped as it is purely an ID column.

**2. Outlier Analysis**

Visualized all numerical columns using boxplots filtered to only show columns with at least one outlier. Ten columns contained outliers including `MonthlyIncome`, `TotalWorkingYears`, and `YearsAtCompany`. All outliers were kept since they represent real employee profiles such as senior executives or long-tenured staff, and XGBoost as a tree-based model is not sensitive to extreme values.

**3. Feature Engineering**

Six features were engineered to capture patterns not directly available in the raw columns. `AvgSatisfaction` averages the four satisfaction scores to measure overall employee happiness. `IncomePerYearExp` flags employees who are underpaid relative to their experience. `StagnationRatio` measures how long an employee has gone without a promotion relative to their tenure. `IsLoyal` flags employees past the five-year mark as low attrition risk. `IsOverworked` combines overtime status with poor work-life balance as a stress signal. `CareerGrowthRate` measures how fast an employee is progressing through job levels. `IsOverworked` made it into the top 5 most important features, validating the engineering decision.

**4. Encoding**

Six binary or ordinal columns were label encoded using a shared mapping dictionary: `Attrition`, `Gender`, and `OverTime` as binary; `BusinessTravel` as ordinal (Non-Travel=0, Rarely=1, Frequently=2); `MaritalStatus` and `Department` as nominal label encoding. `EducationField` and `JobRole` were one-hot encoded using `ColumnTransformer` applied after the train/test split to prevent data leakage.

**5. Class Imbalance**

The dataset is heavily imbalanced with only 16% attrition. Rather than applying SMOTE, `scale_pos_weight` was set to 5.19 (ratio of majority to minority class) directly in XGBoost. This penalizes the model for missing attrition cases without modifying the training data. Recall for class 1 improved from 0.21 to 0.36 after applying this parameter.

**6. Model Selection**

Compared Random Forest and XGBoost baselines. Both models hit a train score of 1.0, indicating overfitting. XGBoost generalized better on the test set with 84% accuracy vs 83% for Random Forest. The classification report confirmed XGBoost had stronger precision and recall on the minority class, making it the chosen model.

**7. Cross Validation**

5-fold stratified cross validation with F1 scoring returned a mean F1 of 0.477 with a standard deviation of 0.018 across all folds, confirming the model is stable and consistent before tuning.

**8. Hyperparameter Tuning**

`GridSearchCV` tested 729 parameter combinations across `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `min_child_weight`. Best parameters were `max_depth=3`, `learning_rate=0.05`, `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=5`. Recall on the minority class improved from 0.36 to 0.57 after tuning.

**9. Threshold Tuning**

Tested thresholds from 0.10 to 0.55. Lowering the threshold from 0.5 to 0.3 increased recall from 0.57 to 0.85, catching 40 out of 47 employees who left in the test set. The tradeoff is more false alarms but for an HR retention tool, missing an at-risk employee is far more costly than a false positive.

## Why Recall Over Accuracy

The dataset is imbalanced with 84% of employees having stayed. A model predicting everyone as staying would achieve 84% accuracy while being completely useless. The real-world cost of a false negative (missing an employee who leaves) is high: recruiting and onboarding replacements can cost months of salary. Recall for the attrition class is the meaningful metric, and threshold tuning was the primary tool used to maximize it.

## Key Decisions

No feature scaling was applied since XGBoost makes decisions through splits rather than distance calculations. `scale_pos_weight` was preferred over SMOTE to handle class imbalance, keeping the training data unmodified. Threshold tuning at 0.30 was chosen over 0.10 despite lower recall because precision at 0.10 collapses to 0.17, meaning HR would receive a mostly useless flagging list with five false alarms for every real leaver.

## Feature Importance (Top 5)

| Feature | Insight |
|---|---|
| YearsAtCompany | Newer employees leave more |
| JobLevel | Junior employees are highest risk |
| OverTime | Overtime is a strong attrition signal |
| StockOptionLevel | Low stock options correlates with leaving |
| IsOverworked | Engineered feature - overtime + poor work-life balance |

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- XGBoost
- seaborn, matplotlib
- joblib

## How to Run

```bash
git clone https://github.com/ammartech-sol/ibm-hr-attrition-classification
cd ibm-hr-attrition-classification
pip install -r requirements.txt
jupyter notebook notebook/notebook.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place the CSV inside the `data/` folder before running the notebook.
