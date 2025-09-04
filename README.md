🌧️ Weather Rain Prediction Project

This project predicts whether it will rain tomorrow using historical weather data from Australia (WeatherAUS dataset). It applies data cleaning, feature engineering, preprocessing, and multiple classification models to achieve high predictive accuracy.

📂 Dataset
File: weatherAUS.csv
Target: RainTomorrow (Yes/No)
Features: Temperature, Humidity, Pressure, Wind speed and direction, Rainfall, Cloud cover, Location, Season, RainToday, and more.

🧹 Data Cleaning & Exploration
Dropped columns with many missing values (Sunshine, Evaporation, Cloud3pm, Cloud9am)
Filled missing categorical values using mode
Imputed numerical features using KNNImputer
Removed outliers using IQR method
Removed duplicate rows
Created exploratory plots: countplots, histograms, boxplots, scatter plots, and a correlation heatmap

🧠 Feature Engineering
Extracted Year and Month from Date
Created a Season column from the month
Calculated WindSpeed_mean from 9am and 3pm values
Mapped Location to average number of rainy days

Encoded features:
Wind direction as angles
Season using OneHotEncoder
RainToday & RainTomorrow using LabelEncoder
Dropped highly correlated and redundant columns

📈 Data Preprocessing
Scaled numerical features using StandardScaler
Addressed class imbalance with SMOTE
Split dataset into training and testing sets (80/20)

🤖 Models Used
Trained and tuned classifiers using GridSearchCV:
Logistic Regression
Random Forest
Gradient Boosting
AdaBoost
Decision Tree
XGBRFClassifier
LightGBM
CatBoost

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report
Most models achieved perfect accuracy and F1 score on the testing set due to careful preprocessing, class balancing, and feature selection. Logistic Regression performed slightly lower. External validation is recommended.

🧰 Libraries Used
pandas, numpy, matplotlib, seaborn
scikit-learn (preprocessing, models, metrics, GridSearchCV)
xgboost, lightgbm, catboost

imblearn (SMOTE)

pickle
