# Yelp_Recommendation_System

## Model Training Process

### Method Used
- XGBoost Regressor (XGBRegressor)

### Hyperparameter Tuning
- Utilized Optuna to train the parameters

### Hybrid Model
- Attempted to incorporate Collaborative Filtering (CF) in a hybrid model, but it resulted in an increase in Root Mean Square Error (RMSE)

### Hyperparameter Search
- Initially used Grid Search CV, but it was too slow, so switched to Optuna for faster optimization

### Optuna Trials
- Conducted 45 trials using Optuna to find the best parameters (45 being the chosen number for trials)

### Data Preprocessing
- Combined compliments due to low counts in each column
- Normalized latitude and longitude, setting them to zero if they didn't exist
- Converted true and false columns to 1 and 0, respectively, for model compatibility
- Categorized 'casual', 'formal', and 'dressy' variables

### Additional Features
- Determined the number of photos for each business and total likes for user/business

## Error Distribution

| Error Range     | Count   |
|-----------------|---------|
| >=0 and <1      | 102,331 |
| >=1 and <2      | 32,787  |
| >=2 and <3      | 6,137   |
| >=3 and <4      | 787     |
| >=4             | 2       |

## Model Evaluation

- **RMSE**: 0.9762416542492776
- **Execution Time**: 451.3598358631134 seconds

