# Error Distribution:                                                             
# >=0 and <1: 102331
# >=1 and <2: 32787
# >=2 and <3: 6137
# >=3 and <4: 787
# >=4: 2

# RMSE: 0.9762416542492776

# Execution Time: 451.3598358631134

from pyspark import SparkContext
import sys
import time
import xgboost
import json
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt 


def objective(trial):
    # defining hyper paramters OPTUNA
    param = {
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    }

    #  fit the model
    model = xgboost.XGBRegressor(**param)
    model.fit(x_train, y_train)

    # predict
    preds = model.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, preds))
    return rmse  



def price(attributes, key):
    if attributes is not None:
        return int(attributes.get(key, 0))
    return 0

def true_false(attributes, key):
    if attributes is not None:
        return 1 if attributes.get(key) == 'True' else 0
    return 0

def convert_attire_to_numeric(attributes, key):
    if attributes is not None:
        attire = attributes.get(key, 'casual').lower()
        return {'casual': 0, 'formal': 1, 'dressy': 2}.get(attire, 0)
    return 0

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def preprocessing(user, business, photo, tip, review):
    partitions = 20
    user_rdd = sc.textFile(user, partitions) \
    .map(lambda line: json.loads(line)) \
    .map(lambda line: (
        line['user_id'],
        (
            float(line['review_count']),
            int(line['fans']),
            float(line['average_stars']),
            int(line['useful']),
            int(line['funny']),
            int(line['cool']),
            int(line['compliment_hot']) +
            int(line['compliment_more']) +
            int(line['compliment_profile']) +
            int(line['compliment_cute']) +
            int(line['compliment_list']) +
            int(line['compliment_note']) +
            int(line['compliment_plain']) +
            int(line['compliment_cool']) +
            int(line['compliment_funny']) +
            int(line['compliment_writer']) +
            int(line['compliment_photos'])
        )
    ))

    
    business_rdd = sc.textFile(business, partitions) \
        .map(lambda line: json.loads(line)) \
        .map(lambda x: (
            x['business_id'],
            (
                float(x['review_count']),
                float(x['stars']),
                int(x['is_open']),
                (safe_float(x["longitude"]) + 180) / 360,  
                (safe_float(x["latitude"]) + 90) / 180,
                true_false(x['attributes'], 'BusinessAcceptsCreditCards'),
                true_false(x['attributes'], 'BikeParking'),
                true_false(x['attributes'], 'GoodForKids'),
                true_false(x['attributes'], 'HasTV'),
                true_false(x['attributes'], 'OutdoorSeating'),
                true_false(x['attributes'], 'RestaurantsGoodForGroups'),
                true_false(x['attributes'], 'RestaurantsDelivery'),
                price(x['attributes'], 'RestaurantsPriceRange2'),
                convert_attire_to_numeric(x['attributes'], 'RestaurantsAttire'),
                true_false(x['attributes'], 'Caters'),
                true_false(x['attributes'], 'RestaurantsReservations'),
                true_false(x['attributes'], 'RestaurantsTableService'),
                true_false(x['attributes'], 'OutdoorSeating'),
                true_false(x['attributes'], 'ByAppointmentOnly'),
                true_false(x['attributes'], 'RestaurantsTakeOut'),
                true_false(x['attributes'], 'AcceptsInsurance'),
                true_false(x['attributes'], 'WheelchairAccessible')
                 
            )
        ))


    photo_rdd = sc.textFile(photo, partitions) \
        .map(lambda line: json.loads(line)) \
        .map(lambda line: (line['business_id'], line['photo_id'])) \
        .groupByKey() \
        .map(lambda line: (line[0], len(line[1])))
    
    tip_rdd = sc.textFile(tip, partitions) \
        .map(lambda line: json.loads(line)) \
        .map(lambda line: ((line['business_id'], line['user_id']), int(line["likes"]))) \
        .reduceByKey(lambda a, b: a + b)

    return user_rdd, business_rdd, photo_rdd, tip_rdd

def partitioning(file):
    partitions = 20
    text_rdd = sc.textFile(file, partitions)
    text_rdd_header = text_rdd.first()
    data_rdd = text_rdd.filter(lambda row: row != text_rdd_header)
    # business_id, user_id, stars
    rdd = data_rdd.map(lambda row: (row.split(',')[0], row.split(',')[1], float(row.split(',')[2])))
    return rdd

def merging_rdds(rdd, user_rdd, business_rdd, photo_rdd, tip_rdd):
    #convert RDDs to dictionaries for efficient lookups
    user_dict = user_rdd.collectAsMap()
    business_dict = business_rdd.collectAsMap()
    photo_dict = photo_rdd.collectAsMap()
    tip_dict = tip_rdd.collectAsMap()
    #define a function to safely extract features and handle None values
    def get_features(user_id, business_id):
        # extract user features or use a default tuple of zeros if user_id is not found
        user_features = user_dict.get(user_id, (np.nan, np.nan, np.nan, np.nan, np.nan,np.nan,np.nan))
        #exxtract business features or use a default tuple of zeros if business_id is not found
        business_features = business_dict.get(business_id, (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        # Get photo count or default to 0 if business_id is not found
        photo_count = photo_dict.get(business_id, np.nan)
        #get tip likes or default to 0 if (user_id, business_id) is not found
        tip_likes = tip_dict.get((user_id, business_id), np.nan)
        #get checkin count or default to 0 if business_id is not found
        
        # Combine all features into a single tuple
        return user_features + business_features + (photo_count, tip_likes)

    # Merge the features with the training RDD
    x_data = rdd.map(lambda x: get_features(x[0], x[1]))
    y_data= rdd.map(lambda x: x[2])

    return x_data,y_data

from collections import Counter

def prediction(x_train, y_train, x_test, y_test,validation_rdd,output_filepath):
    x_train = x_train.collect()
    y_train = y_train.collect()
    x_test = x_test.collect()
    y_test = y_test.collect()

    param = {'lambda': 0.36220341266722017, 'alpha': 0.3207074101012431, 'colsample_bytree': 0.43105334152824387, 'subsample': 0.9588006972426169, 'learning_rate': 0.019090953726479495, 'max_depth': 15, 'min_child_weight': 236, 'n_estimators': 526}

    model = xgboost.XGBRegressor(**param)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # Calculate absolute differences
    abs_diff = np.abs(predictions - y_test)

    # Define the error levels
    error_levels = {
        '>=0 and <1': 0,
        '>=1 and <2': 0,
        '>=2 and <3': 0,
        '>=3 and <4': 0,
        '>=4': 0
    }

    # Categorize each prediction into the error levels
    for diff in abs_diff:
        if diff >= 0 and diff < 1:
            error_levels['>=0 and <1'] += 1
        elif diff >= 1 and diff < 2:
            error_levels['>=1 and <2'] += 1
        elif diff >= 2 and diff < 3:
            error_levels['>=2 and <3'] += 1
        elif diff >= 3 and diff < 4:
            error_levels['>=3 and <4'] += 1
        elif diff >= 4:
            error_levels['>=4'] += 1

    # print the error distribution
    print('Error Distribution:')
    for level, count in error_levels.items():
        print(f"{level}: {count}")

    # print rmse
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print("RMSE:", rmse)

    # now writing the prediciton
    validation_data = validation_rdd.map(lambda x: (x[0], x[1], x[2])).collect()
    with open(output_filepath, 'w') as file:
        file.write('user_id,business_id,prediction\n')
        for i, (user_id, business_id, _) in enumerate(validation_data):
            file.write(f"{user_id},{business_id},{predictions[i]}\n")





if __name__ == '__main__':
    time_start = time.time()
    input_filepath = '/Users/andrewmoore/Desktop/DSCI 553/competition_project'
    validation_filepath = '/Users/andrewmoore/Desktop/DSCI 553/competition_project/yelp_val.csv'
    output_filepath = '/Users/andrewmoore/Desktop/DSCI 553/competition_project/finalized_ouput_final.csv'
    sc = SparkContext('local[*]', 'competition')
    sc.setLogLevel('ERROR')

    user = input_filepath + '/user.json'
    business = input_filepath + '/business.json'
    photo = input_filepath + '/photo.json'
    checkin = input_filepath + '/checkin.json'
    tip = input_filepath + '/tip.json'
    review = input_filepath + '/review_train.json'

    training = input_filepath + '/yelp_train.csv'
    validation = validation_filepath
    training_rdd = partitioning(training)
    validation_rdd = partitioning(validation)

    user_rdd, business_rdd, photo_rdd, tip_rdd = preprocessing(user, business, photo, tip, review)

    x_train,y_train = merging_rdds(training_rdd, user_rdd, business_rdd, photo_rdd, tip_rdd)
    x_test,y_test =  merging_rdds(validation_rdd, user_rdd, business_rdd, photo_rdd, tip_rdd)

    model_prediction = prediction(x_train,y_train,x_test,y_test,validation_rdd,output_filepath)


    #################OPTUNA TRAINING##########################
    # x_train = x_train.collect()
    # y_train = y_train.collect()
    # x_test = x_test.collect()
    # y_test = y_test.collect()

    # # Create a study object and specify the direction is 'minimize'.
    # study = optuna.create_study(direction='minimize')

    # # Optimize the study, the objective function is passed in as the first argument.
    # study.optimize(objective, n_trials=45)  # Specify the number of trials

    # # Output the best parameters
    # print('Number of finished trials:', len(study.trials))
    # print('Best trial:')
    # trial = study.best_trial

    # print('Value:', trial.value)
    # print('Params:')
    # for key, value in trial.params.items():
    #     print(f'    {key}: {value}')

    #################OPTUNA TRAINING##########################



    time_end = time.time()
    duration = time_end - time_start
    final_time = print('Execution Time:', duration)





