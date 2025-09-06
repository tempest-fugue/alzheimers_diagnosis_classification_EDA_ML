# Alzheimer’s Diagnosis Classification: EDA & Machine Learning

This project explores the use of **machine learning models** to classify Alzheimer’s diagnosis outcomes based on structured clinical data. It combines **exploratory data analysis (EDA)**, preprocessing, and benchmarking of multiple algorithms, with the goal of identifying approaches that achieve robust predictive performance.

---

## Project Overview
The notebook walks through the following steps:

1. **Initializations and Downloads**  
   - Import dependencies  
   - Load dataset

2. **Preprocessing**  
   - Data cleaning and feature engineering  
   - Handling missing values  
   - Encoding categorical variables  
   - Train/validation/test splits  

3. **Exploratory Data Analysis (EDA)**  
   - Statistical summaries  
   - Visualizations of feature distributions  
   - Class imbalance checks  

4. **Modeling**  
   Multiple models were trained and compared, including:  
   - Logistic Regression  
   - Random Forest  
   - Extra Trees  
   - Gradient Boosting (LightGBM, XGBoost, CatBoost)  
   - Stochastic Gradient Descent (SGD)  
   - Hyperparameter tuning with GridSearchCV and HalvingRandomSearchCV  

5. **Model Testing**  
   - Evaluation of models on held-out test data  
   - Baseline dummy classifier for comparison  
   - Metrics include accuracy, precision, recall, F1-score, and confusion matrices  

6. **Conclusions**  
   - Key findings from model performance comparisons  
   - Notes on class imbalance and feature importance  
   - Recommendations for future improvements  

---

## Requirements
To reproduce the results, install the following dependencies:

```bash
pip install -r requirements.txt
```
Minimum recommended versions:
- Python 3.9+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- lightgbm
- xgboost
- catboost

## Usage:
Clone this repository:

```bash
git clone https://github.com/yourusername/alzheimers-diagnosis-classification.git
cd alzheimers-diagnosis-classification
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Open the notebook:

```bash
jupyter notebook notebooks/alzheimers_diagnosis_classification_eda_ml.ipynb
```
## Results
Several tree-based ensemble methods (Random Forest, LightGBM, XGBoost, CatBoost) outperformed linear models.

Hyperparameter tuning provided significant performance improvements.

Handling class imbalance (upsampling, downsampling, SMOTE) was essential to improve recall for minority classes.

Detailed results, metrics, and plots are available in the notebook.

Repository Structure
```bash
.
├── notebooks/
│   └── alzheimers_diagnosis_classification_eda_ml.ipynb
├── data/                
├── requirements.txt
└── README.md
```
## Future Work
Incorporate deep learning models (e.g., BERT for clinical notes or tabular transformers).

Perform feature selection and domain-informed feature engineering.

Explore explainability (SHAP, LIME) to interpret predictions.

## License
This project is released under the MIT License. See LICENSE for details.
