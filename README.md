# Yelp_Recommendation_System

## Overview
The project involves a competition to improve the performance of a recommendation system using data mining techniques. Participants are required to enhance the prediction accuracy and efficiency of a recommendation system from a previous assignment (Assignment 3) using any method, such as hybrid recommendation systems.

### Competition Requirements
- **Programming Language**: Python 3.6
- **Library Requirements**: External Python libraries are allowed as long as they are available on Vocareum
- **Spark Usage**: Only Spark RDD is allowed for operations
- **Programming Environment**: Python 3.6, Scala 2.12, JDK 1.8, and Spark 3.1.2

### Yelp Data
The datasets for the competition are provided (some datasets cannot be uploaded due to size), including training and validation datasets, as well as additional information about users, businesses, reviews, check-ins, tips, and photos.

### Task
Participants are tasked with building a recommendation system to predict stars for (user, business) pairs. Evaluation can be done based on error distribution and RMSE.

## Model Training Process

### Method Used
- XGBoost Regressor (XGBRegressor)

### Hyperparameter Tuning
- Utilized Optuna to train the parameters

### Model Selection
- Attempted to incorporate Collaborative Filtering (CF) in a hybrid model, but it resulted in an increase in Root Mean Square Error (RMSE). Instead, I went with a model-based recommendation system.

### Hyperparameter Search
- Initially used Grid Search CV, but it was too slow, so switched to Optuna for faster optimization

### Optuna Trials
- Conducted 45 trials using Optuna to find the best parameters.

### Data Preprocessing
- Combined compliments due to low counts in each column
- Normalized latitude and longitude, setting them to zero if they didn't exist
- Converted true and false columns to 1 and 0, respectively, for model compatibility
- Categorized 'casual', 'formal', and 'dressy' variables
- Determined the number of photos for each business and total likes for user/business


## Error Distribution

| Error Range(stars)    | Count   |
|-----------------|---------|
| >=0 and <1      | 102,331 |
| >=1 and <2      | 32,787  |
| >=2 and <3      | 6,137   |
| >=3 and <4      | 787     |
| >=4             | 2       |

## Model Evaluation

- **RMSE**: 0.9762416542492776
- **Execution Time**: 451.3598358631134 seconds
- **Grade**: 100%

