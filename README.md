# Employee Attrition Prediction
### Hybrid Residual-Ensemble: TabNet + LightGBM

Predicts whether an employee will leave a company using the IBM HR Analytics 
dataset. Compares three models — TabNet, LightGBM, and a Hybrid ensemble — 
with full cross-validation, class imbalance handling, and interpretability analysis.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| TabNet | 62.9% | 0.26 | **0.70** | 0.38 | 0.75 |
| **LightGBM** | **86.0%** | **0.62** | 0.36 | **0.45** | **0.81** |
| Hybrid (TabNet + LightGBM) | 83.9% | — | — | — | 0.79 |

**LightGBM achieves the best overall performance** (86% accuracy, AUC 0.81).

---

## Dataset

- **Source:** [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 employees × 35 features
- **Target:** Attrition — Yes (16.1%) / No (83.9%)
- **Included:** Dataset is already present in the `dataset/` folder — no download needed.

---

## Project Structure

Employee_Attrition_Hybrid_Model/
│
├── src/
│   └── Employee_Attrition_Hybrid_Model.ipynb   ← main notebook
│
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   ← dataset included
│
├── outputs/                                     ← generated plots saved here
│
├── requirements.txt                             ← dependencies
└── README.md



---

## Pipeline

Raw CSV  →  Drop irrelevant columns  →  Encode categoricals
→  Min-Max Scale  →  SMOTE (inside each fold)
→  Stratified 5-Fold CV  →  Train 3 models  →  Compare + Interpret



---

## Models

**TabNet**
- Deep learning model designed for tabular data
- Uses attention mechanism to select important features at each step
- High recall — catches most employees who leave

**LightGBM**
- Gradient boosting tree model
- Best overall accuracy, precision, and AUC
- Fast training with early stopping

**Hybrid Residual Ensemble**
- TabNet trains first and generates probability scores
- LightGBM uses original features + TabNet signal as input
- Aims to correct TabNet errors using gradient boosting

---

## Key Findings

Top drivers of employee attrition:

1. **Overtime** — employees working overtime leave most frequently
2. **Monthly Income** — lower salary = higher attrition risk
3. **Age** — younger employees leave more
4. **Total Working Years** — less experienced employees are higher risk
5. **Years at Company** — newer employees more likely to leave

> **Ethical Check:** Gender and Marital Status rank low in feature importance —
the model does not rely on protected characteristics for predictions.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Vinay2122-dot/Employee_Attrition_Hybrid_Model.git
cd Employee_Attrition_Hybrid_Model
2. Install dependencies


pip install -r requirements.txt
3. Run the notebook


jupyter notebook src/Employee_Attrition_Hybrid_Model.ipynb
Dataset is already included — no extra steps needed.

Tech Stack
Library	Purpose
pytorch-tabnet	TabNet deep learning model
lightgbm	Gradient boosting model
imbalanced-learn	SMOTE for class imbalance
scikit-learn	Preprocessing, CV, metrics
pandas / numpy	Data manipulation
matplotlib / seaborn	Visualizations

Author
Vinay — GitHub


