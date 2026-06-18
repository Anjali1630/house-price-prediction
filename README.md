# House Price Prediction using Machine Learning

## Overview

This project builds a machine learning model to predict house prices using the California Housing dataset from Scikit-learn. The project demonstrates the complete machine learning workflow, including data exploration, visualization, model training, evaluation, and interpretation of results.

The objective is to analyze how factors such as median income, house age, population, and geographic location influence housing prices and use these features to make accurate predictions.

---

## Features

* Exploratory Data Analysis (EDA)
* Data Visualization using Matplotlib and Seaborn
* Correlation Analysis
* Linear Regression Model Training
* Model Performance Evaluation
* Feature Importance Analysis
* Prediction Visualization

---

## Technologies Used

### Programming Language

* Python 3.11+

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Dataset

The project uses the California Housing Dataset provided by Scikit-learn.

### Dataset Statistics

| Attribute       | Value              |
| --------------- | ------------------ |
| Total Samples   | 20,640             |
| Features        | 8                  |
| Target Variable | Median House Value |

### Features Used

* MedInc (Median Income)
* HouseAge (Median House Age)
* AveRooms (Average Rooms)
* AveBedrms (Average Bedrooms)
* Population
* AveOccup (Average Occupancy)
* Latitude
* Longitude

---

## Project Workflow

### 1. Data Collection

Loaded the California Housing dataset using Scikit-learn.

### 2. Exploratory Data Analysis

* Examined dataset structure and statistics
* Identified feature distributions
* Generated correlation heatmaps
* Visualized relationships between variables

### 3. Data Preprocessing

* Checked for missing values
* Prepared features and target variables
* Split dataset into training and testing sets

### 4. Model Development

Implemented a Linear Regression model using Scikit-learn.

### 5. Model Evaluation

Evaluated model performance using:

* R² Score
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

### 6. Result Interpretation

Analyzed feature impact and visualized prediction performance.

---

## Results

### Model Performance

| Metric              | Score                                |
| ------------------- | ------------------------------------ |
| R² Score            | ~0.60                                |
| Mean Squared Error  | Low                                  |
| Prediction Accuracy | Approximately 60% Variance Explained |

### Key Insights

* Median Income is the strongest predictor of house prices.
* Geographic location significantly influences property values.
* Housing age has a moderate impact on pricing.
* Population density contributes to housing demand patterns.

---

## Visualizations

The project includes:

* Correlation Heatmap
* Feature Distribution Plots
* Scatter Plots
* Actual vs Predicted Price Visualization
* Feature Importance Analysis

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/an-ja-li-web/house-price-prediction.git
cd house-price-prediction
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

Run the project:

```bash
python main.py
```

The script will:

1. Load the dataset
2. Perform exploratory analysis
3. Train the model
4. Evaluate performance
5. Display visualizations and results

---

## Project Structure

```text
house-price-prediction/
│
├── main.py
├── README.md
├── requirements.txt
└── images/
    ├── correlation_heatmap.png
    ├── feature_distribution.png
    └── prediction_results.png
```

---

## Skills Demonstrated

* Data Analysis
* Data Visualization
* Statistical Analysis
* Machine Learning
* Predictive Modeling
* Feature Engineering
* Model Evaluation
* Python Programming

---

## Future Improvements

* Implement Random Forest Regressor
* Add XGBoost Regression
* Perform Hyperparameter Tuning
* Add Cross Validation
* Deploy using Streamlit
* Build REST APIs using FastAPI
* Containerize with Docker
* Deploy on Cloud Platforms (AWS/GCP)

---

## Learning Outcomes

Through this project, I gained practical experience in:

* End-to-End Machine Learning Pipelines
* Exploratory Data Analysis (EDA)
* Regression Algorithms
* Model Evaluation Techniques
* Data Visualization
* Predictive Analytics

---

## Author

**Anjali **

Artificial Intelligence & Machine Learning Student

GitHub: https://github.com/Anjali1630


---

## Acknowledgments

* California Housing Dataset from Scikit-learn
* Python Open Source Community
* Scikit-learn Documentation
