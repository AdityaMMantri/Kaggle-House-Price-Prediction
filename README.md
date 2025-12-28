# **Description of the Project**

This repository implements a machine learning solution to predict residential house sale prices using the *House Prices: Advanced Regression Techniques* dataset from Kaggle. The objective is to understand and model the complex relationships between the 80 available features and the final sale price of homes in Ames, Iowa.

This project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training and evaluation, hyperparameter tuning, and generation of Kaggle-ready submissions.

---

## **Key Features and Functionalities**

* **Exploratory Data Analysis (EDA):** Visual analysis of distributions, correlations, and missing data.
* **Data Cleaning & Preprocessing:** Handling missing values, encoding categorical features, target transformation.
* **Feature Engineering:** Creation and transformation of features to improve model performance.
* **Model Training:** Implementation of regression models (e.g., Gradient Boosting, Random Forest, XGBoost).
* **Hyperparameter Optimization:** Grid search or similar tuning strategies to improve model accuracy.
* **Submission Generation:** Produce CSV submission files for the Kaggle competition.
* **Performance Tracking:** Compare multiple models and record results (in `Hyper.xlsx`).

---

## **Installation and Usage Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/AdityaMMantri/Kaggle-House-Price-Prediction.git
cd Kaggle-House-Price-Prediction
```

### **2. Create Python Environment**

Install dependencies (recommended with virtual environment):

```bash
python -m venv venv
source venv/bin/activate       # MacOS/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### **3. Data Setup**

Place the Kaggle competition files (`train.csv`, `test.csv`, `data_description.txt`) inside the appropriate data folder, such as `Pre Processed Dataset` if used for preprocessing.

### **4. Run Notebook**

Open the main notebook to execute all steps:

```bash
jupyter notebook Kaggle_house.ipynb
```

Follow the cells in order:
EDA → Preprocessing → Modeling → Evaluation → Submission.

---

## **Example Outputs or Results** *(fill with your actual metrics)*

| Model            | Validation RMSE | Kaggle Score |
| ---------------- | --------------- | ------------ |
| Net-Elastic+CatBoost+XGboost          | 0.91085           |0.11779     |
| Net-Elastic+CatBoost          | 0.9061          | 0.11991        |


* Example submissions are stored in the `Submissions` folder.
* Hyperparameter tuning records are in `Hyper.xlsx`.

---

## **Acknowledgments or References**

* Kaggle Competition: *House Prices: Advanced Regression Techniques* — [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* Data source: *Ames Housing Dataset* (included within Kaggle competition).
* Inspiration and approaches from similar projects on GitHub and Kaggle discussion forums.
* Scikit-learn documentation for regression modeling.

****
