# Pokemon Classification using Supervised Machine Learning

This project is a modular, production-style machine learning pipeline for classifying Legendary Pokemon using various supervised learning algorithms. It includes detailed Exploratory Data Analysis (EDA), intelligent feature engineering, model benchmarking, and performance visualization. The goal was to deeply understand **how different ML models behave on the same dataset**, and to clearly articulate the reasoning behind each decision.

---

## Objective

Build an end-to-end machine learning pipeline to:

* Analyze Pokemon stats and metadata
* Engineer meaningful features
* Train and evaluate multiple classifiers
* Visualize results and derive insights
* Highlight pros and cons of different model types

---

## Dataset Overview

* Source: Kaggle's [Complete Pokemon Dataset](https://www.kaggle.com/abcsds/pokemon)
* Target: `Legendary` (Boolean)
* Features include:

  * Types (`Type 1`, `Type 2`)
  * Base stats: `HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`
  * Meta: `Generation`, `Name`, `#`

---

## Module Breakdown

### `eda.py` - **DataExplorer**

A class to perform comprehensive EDA on any dataframe.

#### Key Methods:

* `explore_dataframe(df, target=None)`

  * Prints shape, dtypes, missing values, and descriptive stats
  * Plots missing value bars and feature distributions
  * Generates heatmap of correlations
* `_analyze_target(df, target)`

  * Shows class balance and visualizes target variable
* `_plot_distributions(df)`

  * Plots histograms for all numerical features
* `_analyze_categorical_features(df)`

  * Summarizes and plots low-cardinality categorical features

### `feature_engineering.py` - **FeatureEngineer**

Transforms raw data into model-friendly features.

#### Key Methods:

* `create_pokemon_type_features(df)`

  * Converts `Type 1` and `Type 2` into one-hot encoded binary flags
* `create_statistical_features(df)`

  * Adds engineered features:

    * `Total_Stats`, `Avg_Stats`
    * `Attack_Defense_Ratio`, `SpAtk_SpDef_Ratio`
    * `Offensive_Power`, `Defensive_Power`, `Off_Def_Ratio`
* `prepare_pokemon_data(df, target_col='Legendary')`

  * Drops unused columns, applies all transformations, and returns train/test splits

### `model_trainer.py` - **ModelTrainer**

Handles model training, evaluation, and visualization.

#### Key Methods:

* `get_model_configs()`

  * Returns a dictionary of ML models with default parameters
* `train_and_evaluate_models(X_train, X_test, y_train, y_test)`

  * Trains models and evaluates using:

    * Accuracy, Precision, Recall, F1 Score, Cross-validation
* `_calculate_metrics(...)`

  * Computes evaluation metrics per model
* `optimize_knn(...)`

  * Performs GridSearchCV to optimize KNN parameters
* `visualize_results(results_df)`

  * Creates side-by-side bar plots for all metrics
* `generate_classification_report(X_test, y_test, model_name)`

  * Prints sklearn classification report + plots confusion matrix
* `get_feature_importance(model_name, feature_names)`

  * Visualizes top 15 important features for tree-based models

---

## Model Overview

### Logistic Regression

* **Type:** Linear classifier
* **Why Used:** Baseline model, interpretable coefficients
* **Strengths:** Fast, low-variance, works well with scaled data
* **Weaknesses:** Poor with non-linearly separable data

### Decision Trees (Gini, Entropy)

* **Type:** Tree-based non-linear classifier
* **Why Used:** Captures complex relationships, no need for feature scaling
* **Strengths:** Interpretable, handles categorical/numeric data
* **Weaknesses:** High variance, prone to overfitting without tuning

### Random Forest

* **Type:** Ensemble of decision trees
* **Why Used:** Combats overfitting, handles complex feature interactions
* **Strengths:** High accuracy, robust to noise
* **Weaknesses:** Less interpretable, slower to train

### K-Nearest Neighbors (KNN)

* **Type:** Instance-based learner
* **Why Used:** Non-parametric model good for pattern detection
* **Strengths:** Simple, good for well-clustered data
* **Weaknesses:** Slow prediction, sensitive to scaling and outliers

### Naive Bayes

* **Type:** Probabilistic classifier
* **Why Used:** Lightweight, good with text/data with independence assumption
* **Strengths:** Fast, good with high-dimensional data
* **Weaknesses:** Assumes feature independence, which may not hold

### Support Vector Machines (Linear & RBF)

* **Type:** Maximum margin classifiers
* **Why Used:** Strong performance with clear decision boundaries
* **Strengths:** Effective in high dimensions, robust to overfitting
* **Weaknesses:** Sensitive to parameters, requires scaling

---

## Setup Instructions

```bash
git clone https://github.com/yourusername/pokemon-ml-classifier.git
cd pokemon-ml-classifier
pip install -r requirements.txt
jupyter notebook notebooks/pokemon_classification.ipynb
```

---

## Final Results (Test Set)

| Model                | Accuracy   | Precision  | Recall     | F1 Score   | CV Accuracy |
| -------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| Logistic Regression  | 0.9125     | 0.8935     | 0.9125     | 0.9003     | 0.9516      |
| Decision Tree (Gini) | 0.9500     | 0.9539     | 0.9500     | 0.9516     | 0.9375      |
| Decision Tree (Ent)  | 0.9438     | 0.9555     | 0.9438     | 0.9478     | 0.9469      |
| K-NN                 | (Tuned)    | N/A        | N/A        | N/A        | N/A         |
| Random Forest        | **0.9688** | **0.9700** | **0.9688** | **0.9693** | **0.9641**  |
| Naive Bayes          | 0.7563     | 0.8953     | 0.7563     | 0.8072     | 0.7719      |
| SVC (Linear)         | 0.9250     | 0.9199     | 0.9250     | 0.9221     | 0.9563      |
| SVC (RBF)            | 0.9313     | 0.9208     | 0.9313     | 0.9141     | 0.9453      |

---

> ✨ If you liked this project, feel free to give it a star and check out more of my work on GitHub!