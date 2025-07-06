 House Price Prediction Project
A machine learning project that predicts house prices using the California Housing dataset. Built with Python, scikit-learn, and data visualization libraries.
 Project Overview
This project demonstrates fundamental machine learning concepts by predicting house prices based on various features like median income, house age, and geographic location.
Features

Data Exploration: Comprehensive analysis of the California Housing dataset
Data Visualization: Interactive charts and correlation analysis
Machine Learning: Linear regression model for price prediction
Model Evaluation: Performance metrics and visualization
Feature Importance: Analysis of which factors most affect house prices

Technologies Used

Python 3.11+
pandas - Data manipulation and analysis
numpy - Numerical computing
matplotlib - Data visualization
seaborn - Statistical data visualization
scikit-learn - Machine learning library

 Prerequisites
Make sure you have Python 3.11+ installed on your system.
 Installation
Clone this repository:
bashgit clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
Install required packages:
bashpip install pandas numpy matplotlib seaborn scikit-learn
 Usage
Run the main script:
bashpython main.py
 Results
The model achieves:
R² Score: ~0.6 (60% accuracy)
Mean Squared Error: Low prediction error
Key Insights: Median income is the strongest predictor of house prices
 Dataset Information
The California Housing dataset contains:
20,640 samples
8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
Target: Median house value (in hundreds of thousands of dollars)
 Key Findings
Median Income has the strongest correlation with house prices
Geographic location (Latitude/Longitude) significantly impacts prices
House age has a moderate negative correlation with prices
Population density affects pricing in urban areas
📁 Project Structure
house-price-prediction/
│
├── main.py                 
├── README.md              
└── requirements.txt       
 Learning Outcomes
This project demonstrates:

Data loading and preprocessing
Exploratory data analysis (EDA)
Feature correlation analysis
Linear regression implementation
Model evaluation and validation
Data visualization techniques

Future Enhancements

 Add more advanced ML models (Random Forest, XGBoost)
 Implement feature engineering
 Add cross-validation
 Create a web interface
 Add model deployment capabilities

Contributing
Feel free to fork this project and submit pull requests for any improvements.
 License
This project is open source and available under the MIT License.
 Author
anjali

GitHub: an-ja-li-web
 Acknowledgments
California Housing dataset from scikit-learn
Python community for excellent libraries
AI/ML community for inspiration