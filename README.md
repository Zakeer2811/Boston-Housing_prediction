# Boston-Housing_prediction
This project predicts housing prices using a Random Forest Regressor. It features data preprocessing, hyperparameter tuning, and evaluation with R² scores. The model includes feature importance analysis and a learning curve, allowing for user input predictions.
Here’s an updated README.md file that includes the exact R² score and the features along with their labels based on your output:

```markdown
# Housing Price Prediction using Random Forest

This project predicts housing prices using a Random Forest Regressor. It includes data preprocessing, hyperparameter tuning, and evaluation using an R² score of **0.888**. The model provides feature importance analysis and visualizes learning curves, allowing users to input data for real-time predictions.

## Table of Contents

- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Training](#model-training)
- [Features](#features)
- [Results](#results)

## Getting Started

To run this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
pip install -r requirements.txt
```

## Dependencies

- pandas
- scikit-learn
- matplotlib

## Usage

1. Place your `HousingData.csv` dataset in the root directory of the project.
2. Run the main script:

```bash
python Bos_house_pred.py
```

3. Follow the prompts to enter feature values for prediction.

## Model Training

The model uses a Random Forest Regressor with hyperparameter tuning through GridSearchCV. Important hyperparameters include:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

## Features

The dataset includes the following features that influence housing prices:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: \(1000(Bk - 0.63)^2\) where \(Bk\) is the proportion of Black residents by town
- **LSTAT**: Percentage of lower status of the population

## Results

The model is evaluated using the R² score, which indicates its predictive performance (R² score: **0.888**). Feature importances are saved to a CSV file for further analysis, and a learning curve is plotted to visualize training progress. The feature importances are as follows:

| Feature | Importance |
|---------|------------|
| RM      | 0.560640   |
| LSTAT   | 0.243393   |
| DIS     | 0.060263   |
| CRIM    | 0.043646   |
| PTRATIO | 0.018016   |
| TAX     | 0.016965   |
| NOX     | 0.016219   |
| AGE     | 0.013350   |
| B       | 0.013335   |
| INDUS   | 0.006537   |
| RAD     | 0.004731   |
| ZN      | 0.001730   |
| CHAS    | 0.001176   |

```
