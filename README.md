# IBM HR Attrition Prediction

A binary classification project to predict whether an employee is likely to leave the company, using the IBM HR Analytics dataset. The end goal is an early warning system for HR teams, not just a model that scores well on paper.

---

## Results

**Final Model: XGBoost Classifier (Threshold = 0.30)**

| Metric | Default Threshold (0.5) | Tuned Threshold (0.30) |
|---|---|---|
| Recall (Attrition) | 0.57 | **0.85** |
| Precision (Attrition) | 0.41 | 0.28 |
| F1 (Attrition) | 0.48 | 0.42 |
| Accuracy | 80% | 63% |

The model catches **40 out of 47** employees who actually left in the test set. Accuracy dropped from 80% to 63% after threshold tuning and that is completely fine. On a dataset where 84% of employees stayed, accuracy is the wrong thing to look at.

---

## Dataset

- **Source:** [IBM HR Analytics on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 rows, 35 features
- **Target:** `Attrition` - `Yes` (left) or `No` (stayed)
- **Class split:** 84% stayed, 16% left

---

## Folder Structure

```
ibm-hr-attrition/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/
│   ├── xgboost_ibm_attrition.pkl
│   ├── ohe_encoder.pkl
│   ├── threshold.pkl
│   └── attrition_columns.pkl
├── notebook/
│   └── notebook.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Project Walkthrough

### 1. Data Cleaning

No missing values anywhere. Three columns got dropped immediately: `EmployeeCount`, `StandardHours`, and `Over18` all had the exact same value across all 1,470 rows - zero variance, zero information. `EmployeeNumber` was also dropped since it is just a row ID.

### 2. Outlier Analysis

Every numerical column was checked using the IQR method and only columns with at least one outlier were visualized. Ten columns had outliers, including `MonthlyIncome`, `TotalWorkingYears`, and `YearsAtCompany`. None were removed. These are real employee profiles - a VP with 30 years of experience is not a data error, and XGBoost does not care about outliers anyway since it splits on values rather than distances.

### 3. Feature Engineering

Six features were added on top of the original 35:

- **`AvgSatisfaction`** - Average of the four satisfaction scores (job, environment, relationship, work-life balance). One low score might be noise. Low averages across all four usually are not.
- **`IncomePerYearExp`** - Monthly income divided by total working years. Catches employees who are being paid below what their experience would suggest.
- **`StagnationRatio`** - Years since last promotion divided by years at company. A high ratio means someone has been sitting in the same role for a long time relative to how long they have been there.
- **`IsLoyal`** - Binary flag for employees past five years. Attrition risk drops significantly after this point.
- **`IsOverworked`** - Flags employees who are on overtime AND have a work-life balance score of 2 or below. Overtime alone does not predict attrition well. Combined with poor work-life balance, it does. This feature ranked in the **top 5 out of all 35 original features plus the engineered ones**.
- **`CareerGrowthRate`** - Job level divided by total working years. Measures how fast someone is moving up relative to how long they have been working.

### 4. Encoding

A shared mapping dictionary was used for six columns: `Attrition` and `OverTime` as binary (No=0, Yes=1); `Gender` (Female=0, Male=1); `BusinessTravel` as ordinal (Non-Travel=0, Rarely=1, Frequently=2); `MaritalStatus` and `Department` as nominal label encoding. `EducationField` and `JobRole` were one-hot encoded using `ColumnTransformer`. The encoder was fit on training data only and then applied to the test set - not the other way around.

### 5. Class Imbalance

Only 16% of the dataset is the positive class. `scale_pos_weight` was set to 5.19, which is the ratio of majority to minority samples in the training set. This tells XGBoost to penalize missing an attrition case roughly five times more than missing a non-attrition case. SMOTE was not used because it generates synthetic samples and adds noise to the training distribution. `scale_pos_weight` keeps the training data exactly as it is and adjusts the loss function instead. Recall on the minority class went from 0.21 to 0.36 just from this one change.

### 6. Model Selection

Random Forest and XGBoost were both trained as baselines. Both hit a train score of 1.0, which means both overfit completely. On the test set, XGBoost came out at 84% accuracy vs 83% for Random Forest. More importantly, XGBoost had better precision and recall on the minority class across the board. XGBoost moved forward.

### 7. Cross Validation

5-fold stratified cross-validation scored on F1 returned a mean of **0.477** with a standard deviation of **0.018**. Low variance across folds means the model is not getting lucky on any particular split.

### 8. Hyperparameter Tuning

`GridSearchCV` ran over 729 combinations testing `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `min_child_weight`. Scoring was set to recall to stay focused on catching leavers.

Best parameters found:

| Parameter | Value |
|---|---|
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `n_estimators` | 100 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `min_child_weight` | 5 |

Minority class recall went from 0.36 to 0.57 after tuning. Useful, but not the biggest gain.

### 9. Threshold Tuning

Thresholds from 0.10 to 0.55 were swept in 0.05 increments. Lowering from 0.50 to 0.30 brought recall up from 0.57 to 0.85 - that is a bigger jump than tuning gave. The threshold was not pushed below 0.30 because precision at 0.10 collapses to 0.17. An HR team getting a list that is wrong 83% of the time is not going to use the model.

---

## Why Accuracy Is the Wrong Metric Here

If a model predicted "stayed" for every single employee, it would get 84% accuracy on this dataset. That model is useless. It catches zero people who are actually leaving.

Replacing an employee who quits costs months of recruiting, interviewing, and onboarding time. A false positive - flagging someone who was not actually going to leave - costs a retention conversation that did not need to happen. The asymmetry is obvious. Recall is the metric that maps to real cost.

---

## Feature Importance (Top 5)

| Feature | Insight |
|---|---|
| `YearsAtCompany` | The early years are the highest risk window |
| `JobLevel` | Junior employees leave at much higher rates |
| `OverTime` | Strong standalone signal even before engineering |
| `StockOptionLevel` | Low equity correlates with leaving |
| `IsOverworked` | Engineered feature - overtime combined with poor work-life balance |

The highest-risk employee profile that came out of this: **junior level, working overtime, low or no stock options, under five years at the company**.

---

## Saved Files

| File | Description |
|---|---|
| `models/xgboost_ibm_attrition.pkl` | Trained XGBoost classifier |
| `models/ohe_encoder.pkl` | Fitted ColumnTransformer for preprocessing |
| `models/threshold.pkl` | Final classification threshold (0.30) |
| `models/attrition_columns.pkl` | Ordered feature columns expected by the model |
| `notebook/notebook.ipynb` | Full notebook with code and outputs |

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

---

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/ammartech-sol/ibm-hr-attrition-classification
cd ibm-hr-attrition-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The dataset is already in the `data/` folder. Open the notebook and run from top to bottom:
```bash
jupyter notebook notebook/notebook.ipynb
```

---

## Key Takeaways

- An 84% accurate model that catches zero leavers is worse than useless - always check what the majority-class baseline accuracy is before trusting any number
- `scale_pos_weight` is cleaner than SMOTE for this kind of imbalance because it does not touch the training data
- Threshold tuning gave more recall improvement than GridSearchCV did
- `class_weight='balanced'` was not tested here since XGBoost handles imbalance natively through `scale_pos_weight`, but it is worth testing on any sklearn model before assuming it helps
- Fit encoders on training data, transform test data. Every time, no exceptions.
- `IsOverworked` making the top 5 out of 41 total features shows that combining two weak signals can produce a stronger one than either alone
