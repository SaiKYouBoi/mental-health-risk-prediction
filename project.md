# Student Mental Health Risk Prediction — Data Science Project

## 1. Project Context

An educational institution aims to analyze a student dataset to investigate the factors associated with well-being and mental health risks, and then develop a supervised classification prototype.

The project covers the entire Data Science workflow, including:

- Data auditing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing
- Model training
- Hyperparameter optimization
- Model evaluation
- Pipeline automation
- Data visualization

Important Note: This project is strictly for educational purposes. The model must never be presented as a diagnostic tool, a clinical screening tool, or a system for making disciplinary decisions or selecting students.

Dataset: [Student Mental Health Risk Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students?utm_source=chatgpt.com)

## 2. Business Objectives

The main objectives of this project are:

1. Risk Identification: Identify students who may have an increased risk of experiencing psychological difficulties.
2. Prevention: Anticipate potential needs for psychological and emotional support.
3. Factor Analysis: Understand which factors are associated with students' mental health risks.
4. Student Support: Help guide students toward appropriate support resources.
5. Resource Optimization: Prioritize student support initiatives and improve the allocation of available resources.

## 3. Project Implementation Steps

### Step 1 — Data Loading and Exploratory Data Analysis (EDA)

Objective: Understand the dataset, assess its quality, and explore the relationships between student characteristics and mental health risk levels.

Tasks:

1\. Understand the dataset structure

- Identify the number of rows and columns.
- Examine variable names and data types.
- Display the first few records to understand the data.

2\. Assess data quality

- Check for missing values.
- Identify duplicate records.
- Detect inconsistent values.
- Identify potential outliers.

3\. Generate descriptive statistics

- Calculate mean, median, standard deviation, minimum, and maximum values.
- Examine frequency distributions of categorical variables.

4\. Analyze the target variable

- Study the distribution of mental health risk categories.
- Determine whether the dataset suffers from class imbalance.

5\. Explore relationships between features and the target

- Investigate how different student characteristics relate to mental health risk.
- Use appropriate visualizations to identify patterns and trends.

6\. Perform correlation analysis

- Examine correlations between numerical features.
- Identify potential relationships and multicollinearity.

7\. Identify important features

- Determine which variables are most strongly associated with mental health risk levels.

### Step 2 — Data Transformation and Preprocessing

Objective: Prepare a clean and consistent dataset for machine learning while preventing data leakage.

Tasks:

1\. Split the dataset

- Training set: 80%
- Test set: 20%

2\. Handle missing values

- Choose appropriate imputation techniques based on variable types.
- For example, mean or median imputation for numerical variables and mode imputation for categorical variables.

3\. Detect and handle outliers

- Apply suitable methods such as:
  - Interquartile Range (IQR)
  - Z-score
- Decide whether to remove, cap, or retain outliers based on the nature of the data.

4\. Encode categorical variables

- Convert categorical features into numerical representations.
- Use techniques such as One-Hot Encoding or Ordinal Encoding, depending on the variable.

5\. Handle class imbalance

- Evaluate whether class balancing is necessary.
- Apply techniques such as:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Random oversampling
  - Random undersampling

6\. Scale numerical features

- Apply normalization or standardization where appropriate.
- Possible techniques include:
  - StandardScaler
  - MinMaxScaler

7\. Build a preprocessing pipeline

- Combine preprocessing transformations into a single pipeline.
- Ensure preprocessing steps are fitted only on training data.
- Prevent data leakage between training and test datasets.

Important Requirement: Every data cleaning and preprocessing decision must be clearly justified.

### Step 3 — Model Training and Comparison

Objective: Train and compare several supervised machine learning models to identify promising approaches for predicting mental health risk categories.

Tasks:

1\. Train at least three classification models.

Possible models include:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

2\. Ensure a fair comparison

- Use the same training dataset for all models.
- Apply a consistent preprocessing and validation strategy.

3\. Evaluate model performance

&#x20;  Use appropriate classification metrics:

- Accuracy: Overall proportion of correct predictions.
- Precision: Proportion of predicted positive cases that are actually positive.
- Recall: Proportion of actual positive cases correctly identified.
- F1-score: Harmonic mean of precision and recall.
- Confusion Matrix: Visualization of correct and incorrect predictions across classes.
- ROC-AUC: Measure of the model's ability to distinguish between classes, when appropriate.

4\. Compare the results

- Summarize model performances in a comparison table.
- Identify strengths and weaknesses of each approach.

### Step 4 — Hyperparameter Optimization and Final Evaluation

Objective: Improve the most promising models, select the most appropriate one, and evaluate its generalization performance.

Tasks:

1\. Select the most promising models

- Identify the best-performing candidates from the initial comparison.

2\. Perform hyperparameter tuning

- Use optimization techniques such as:
  - GridSearchCV
  - RandomizedSearchCV
- Optimize model parameters using cross-validation on the training dataset.

3\. Compare optimized models

- Evaluate improvements resulting from hyperparameter tuning.
- Compare the performance and complexity of the models.

4\. Select the final model

- Choose the most suitable model based on:
  - Predictive performance
  - Generalization ability
  - Interpretability
  - Business context and project objectives

5\. Evaluate the final model on the independent test set

- Calculate the selected evaluation metrics.
- Generate a confusion matrix.
- Assess generalization to unseen data.

6\. Analyze prediction errors

- Examine false positives and false negatives.
- Discuss the consequences of different types of errors.
- Identify potential bias, overfitting, and limitations.

7\. Present and justify the final model

- Summarize the results.
- Explain why the selected model is the most appropriate for the project.

8\. Save the complete machine learning pipeline

- Save both preprocessing steps and the trained model.
- Use one of the following formats:
  - `.pkl`
  - `.joblib`

### Step 5 — Workflow Orchestration with Apache Airflow and Streamlit Application

Objective: Automate the machine learning workflow and develop an interactive application to visualize model performance and demonstrate predictions.

#### Part A — Workflow Orchestration with Apache Airflow

Design an Apache Airflow DAG (Directed Acyclic Graph) to automate the machine learning pipeline.

Tasks:

1\. Create a DAG that orchestrates the following stages:

- Data preparation
- Data preprocessing
- Model training
- Model evaluation

2\. Configure scheduled DAG execution.

3\. Monitor pipeline execution:

- Track task status.
- Identify failures.
- Monitor execution errors.
- Review execution logs.

#### Part B — Interactive Application with Streamlit

Develop a Streamlit web application that provides an interactive interface for exploring and demonstrating the machine learning model.

Required features:

1\. Model Performance Dashboard

- Display evaluation metrics such as accuracy, precision, recall, and F1-score.
- Present relevant performance charts.
- Visualize the confusion matrix.

2\. Interactive Prediction Interface

- Allow users to enter new student data.
- Apply the saved preprocessing pipeline.
- Generate predictions using the trained classification model.
- Display the predicted risk category.

3\. Model Results Visualization

- Present predictions and performance results in a clear and understandable format.
- Help users interpret the model's output.

4\. Responsible Use

- Clearly indicate that the predictions are educational demonstrations, not clinical assessments.
- Avoid presenting predicted risk categories as medical diagnoses or as a basis for decisions affecting individual students.

## 4. Expected Project Deliverables

Based on the project requirements, the expected deliverables can be organized as follows:

| Project Stage           | Expected Deliverable                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| Step 1 — EDA            | Dataset audit, descriptive statistics, and exploratory visualizations                         |
| Step 2 — Preprocessing  | Cleaned data workflow and reusable preprocessing pipeline                                     |
| Step 3 — Model Training | At least three trained classification models and a performance comparison                     |
| Step 4 — Optimization   | Hyperparameter tuning results, final model evaluation, and saved `.pkl` or `.joblib` pipeline |
| Step 5 — Airflow        | Automated, scheduled DAG with execution monitoring                                            |
| Step 5 — Streamlit      | Interactive dashboard with performance metrics, confusion matrix, and prediction interface    |

## 5. Overall Project Workflow

Student Mental Health Dataset

Data loading and auditing

Exploratory Data Analysis

Statistics, correlations, visualizations

Preprocessing Pipeline

Split, imputation, encoding, scaling, balancing

Model Training and Comparison

At least three classifiers

Optimization and Evaluation

Cross-validation, tuning, independent testing

Saved Machine Learning Pipeline

Preprocessing + final classifier (.joblib / .pkl)

Apache Airflow

Workflow orchestration and automation

Streamlit

Interactive dashboard and prediction demo

Final Project Goal: Develop an end-to-end, reproducible, and automated machine learning prototype that analyzes factors associated with student mental health risks, compares classification algorithms, and presents results through an interactive Streamlit application, while maintaining clear ethical limitations on its use.