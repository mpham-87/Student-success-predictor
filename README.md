# Student Success Predictor

A machine learning project that predicts whether a student may need additional academic support based on academic habits and support-related factors.

The goal of this project is to explore how machine learning can be used to identify patterns associated with lower academic performance and provide an early-warning style prediction that could help educators identify students who may benefit from additional support.

## Overview

This project uses student performance data to train a classification model that predicts whether a student is at risk of receiving a final grade below 10.

The model uses features available before the final grade, including:

- Study time
- Number of absences
- Previous course failures
- School-provided academic support
- Family educational support
- Participation in extracurricular activities
- Interest in pursuing higher education

The final prediction is:

- `0` = Not flagged for additional support
- `1` = May need additional academic support

This project is intended as a demonstration of machine learning and predictive modeling. It is not intended to replace educator judgment or make decisions about students.

## Machine Learning Approach

The project uses a **Random Forest Classifier** implemented with scikit-learn.

The workflow includes:

1. Loading and validating the dataset
2. Selecting relevant features
3. Handling missing values
4. Encoding categorical variables
5. Scaling numeric features
6. Splitting the dataset into training and testing sets
7. Training a Random Forest classifier
8. Evaluating model performance
9. Applying 5-fold stratified cross-validation
10. Performing hyperparameter tuning with GridSearchCV
11. Saving the trained model for use in the demo application

## Dataset

This project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

The dataset contains information about secondary school students, including academic, social, and school-related features.

Dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

This project uses:

```text
student-por.csv